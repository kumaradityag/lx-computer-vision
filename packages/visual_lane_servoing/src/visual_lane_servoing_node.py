#!/usr/bin/env python3

import os
import cv2
import yaml
from typing import Union
import time
import rospy
import numpy as np

from dt_computer_vision.camera import CameraModel
from dt_computer_vision.ground_projection import GroundProjector
from dt_computer_vision.camera.homography import Homography, HomographyToolkit

from turbojpeg import TurboJPEG
from duckietown_msgs.msg import Twist2DStamped
from sensor_msgs.msg import CompressedImage, CameraInfo
from std_msgs.msg import String

from visual_lane_servoing.include import visual_servoing_solution
from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown.utils.image.ros import (
    compressed_imgmsg_to_rgb,
    rgb_to_compressed_imgmsg,
)


class LaneServoingNode(DTROS):
    """
    Performs a form of visual servoing based on estimates of the image-space lane orientation
    Args:
        node_name (:obj:`str`): a unique, descriptive name for the ROS node
    Configuration:

    Publisher:
        ~wheels_cmd (:obj:`WheelsCmdStamped`): The corresponding resulting wheel commands
    Subscribers:
        ~/image/compressed (:obj:`CompressedImage`):
            compressed image
    """

    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(LaneServoingNode, self).__init__(
            node_name=node_name, node_type=NodeType.LOCALIZATION
        )
        self.loginfo("Initializing...")
        # get the name of the robot
        self.veh = rospy.get_namespace().strip("/")

        self.v_0 = 0.2  # Forward velocity command

        # The following are used for scaling
        self.steer_max = -1

        self.VLS_ACTION = None
        self.VLS_STOPPED = True

        # Defining subscribers:
        rospy.Subscriber(
            f"/{self.veh}/camera_node/image/compressed",
            CompressedImage,
            self.cb_image,
            buff_size=10000000,
            queue_size=1,
        )

        self.sub_camera_info = rospy.Subscriber(
            f"/{self.veh}/camera_node/camera_info",
            CameraInfo,
            self.cb_info,
            queue_size=1,
        )

        self.camera_model = None
        self.mapx = None
        self.mapy = None
        self.jpeg = TurboJPEG()

        # select the current activity
        rospy.Subscriber(
            f"/{self.veh}/vls_node/action", String, self.cb_action, queue_size=1
        )

        # Command publisher
        car_cmd_topic = f"/{self.veh}/joy_mapper_node/car_cmd"
        self.pub_car_cmd = rospy.Publisher(
            car_cmd_topic, Twist2DStamped, queue_size=1, dt_topic_type=TopicType.CONTROL
        )

        self._lt_mask_pub = rospy.Publisher(
            f"/{self.veh}/visual_control/left_mask/image/compressed",
            CompressedImage,
            queue_size=1,
        )

        self._rt_mask_pub = rospy.Publisher(
            f"/{self.veh}/visual_control/right_mask/image/compressed",
            CompressedImage,
            queue_size=1,
        )

        # Get the steering gain (omega_max) from the calibration file
        # It defines the maximum omega used to scale normalized steering command
        kinematics_calib = self.read_params_from_calibration_file()
        self.omega_max = kinematics_calib.get("omega_max", 2.0)

        self.loginfo("Initialized!")

    def cb_info(self, msg):
        self.loginfo(
            "Camera info message received. Unsubscribing from camera_info topic."
        )
        try:
            self.sub_camera_info.shutdown()
        except BaseException:
            pass
        H, W = msg.height, msg.width
        # create new camera info
        self.camera_model = CameraModel(
            width=W,
            height=H,
            K=np.reshape(msg.K, (3, 3)),
            D=np.reshape(msg.D, (5,)),
            P=np.reshape(msg.P, (3, 4)),
        )
        homography = self.load_extrinsics()
        self.camera_model.H = homography
        self.projector = GroundProjector(self.camera_model)

        rect_camera_K, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_model.K, self.camera_model.D, (W, H), alpha=0.0
        )
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(
            self.camera_model.K,
            self.camera_model.D,
            None,
            rect_camera_K,
            (W, H),
            cv2.CV_32FC1,
        )

    def cb_action(self, msg):
        """
        Call the right functions according to desktop icon the parameter.
        """

        if msg.data not in ["init", "calibration", "go", "stop"]:
            self.loginfo(f"Activity '{msg.data}' not recognized. Exiting...")
            exit(1)

        self.VLS_ACTION = msg.data

        self.loginfo(f"ACTION: {self.VLS_ACTION}")

        if self.VLS_ACTION == "init":
            self.loginfo("Put the robot in a lane. Press [Calibrate] when done.")
            return

        if self.VLS_ACTION == "calibration":
            self.loginfo(
                "Using your hands if you are working with a real robot, or the joystick if "
                "you are working with the simulator. Turn the robot (in place), to the left "
                "then to the right by about 30deg on each side. Press [Go] when done."
            )
            return

        if self.VLS_ACTION == "go":
            self.loginfo(f"Calibration value: {int(self.steer_max)}")
            self.VLS_STOPPED = False
            # NOTE: this is needed to trigger the agent and get another image back
            self.publish_command([0, 0])
            return

        if self.VLS_ACTION == "stop":
            self.publish_command([0, 0])
            self.VLS_STOPPED = True
            return

    def cb_image(self, image_msg):
        """
        Processes the incoming image messages.

        Performs the following steps for each incoming image:

        #. Resizes the image to the ``~img_size`` resolution
        #. Removes some amount of the image

        Args:
            image_msg (:obj:`sensor_msgs.msg.CompressedImage`): The received image message

        """

        # make sure we have a map to use
        if self.mapx is None or self.mapy is None:
            self.loginfo("Waiting for Camera Info")
            return

        dist_img = self.jpeg.decode(image_msg.data)
        image = cv2.remap(dist_img, self.mapx, self.mapy, cv2.INTER_NEAREST)
        # Resize the image to the desired dimensionsS
        height_original, width_original = image.shape[0:2]
        img_size = image.shape[0:2]
        if img_size[0] != width_original or img_size[1] != height_original:
            image = cv2.resize(
                image, tuple(reversed(img_size)), interpolation=cv2.INTER_NEAREST
            )

        if self.is_shutdown:
            self.publish_command([0, 0])
            return

        shape = image.shape[0:2]

        steer_matrix_left_lm = (
            visual_servoing_solution.get_steer_matrix_left_lane_markings(shape)
        )
        steer_matrix_right_lm = (
            visual_servoing_solution.get_steer_matrix_right_lane_markings(shape)
        )

        # Call the user-defined function to get the masks for the left
        # and right lane markings
        (lt_mask, rt_mask) = visual_servoing_solution.detect_lane_markings(
            image, self.projector
        )

        # Publish these out for visualization
        lt_mask_viz = cv2.addWeighted(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            0.1,
            lt_mask.astype(np.uint8),
            0.8,
            0,
        )
        rt_mask_viz = cv2.addWeighted(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            0.1,
            rt_mask.astype(np.uint8),
            0.8,
            0,
        )

        lt_mask_viz = rgb_to_compressed_imgmsg(
            cv2.cvtColor(lt_mask_viz, cv2.COLOR_GRAY2RGB), "jpeg"
        )
        rt_mask_viz = rgb_to_compressed_imgmsg(
            cv2.cvtColor(rt_mask_viz, cv2.COLOR_GRAY2RGB), "jpeg"
        )

        self._lt_mask_pub.publish(lt_mask_viz)
        self._rt_mask_pub.publish(rt_mask_viz)

        if self.VLS_ACTION == "calibration":
            self.steer_max = max(
                self.steer_max,
                2
                * max(
                    float(np.sum(lt_mask * steer_matrix_left_lm)),
                    float(np.sum(rt_mask * steer_matrix_right_lm)),
                ),
            )

        if self.VLS_ACTION != "go" or self.VLS_STOPPED:
            return

        if self.steer_max == -1:
            self.logerr("Not Calibrated!")
            return

        steer = float(np.sum(lt_mask * steer_matrix_left_lm)) + float(
            np.sum(rt_mask * steer_matrix_right_lm)
        )

        # now rescale from 0 to 1
        steer_scaled = np.sign(steer) * rescale(
            min(np.abs(steer), self.steer_max), 0, self.steer_max
        )

        u = [self.v_0, steer_scaled * self.omega_max]
        self.publish_command(u)

        # self.logging to screen for debugging purposes
        self.loginfo("    VISUAL SERVOING    ")
        self.loginfo(
            f"Steering: (Unnormalized) : {int(steer)} / {int(self.steer_max)},"
            f"  Steering (Normalized) : {np.round(steer_scaled, 1)}"
        )
        self.loginfo(f"Command v : {np.round(u[0], 2)},  omega : {np.round(u[1], 2)}")

    def publish_command(self, u):
        """Publishes a car command message.

        Args:
            u (:obj:`tuple(double, double)`): tuple containing [v, w] for the control action.
        """

        car_control_msg = Twist2DStamped()
        car_control_msg.header.stamp = rospy.Time.now()

        car_control_msg.v = u[0]  # v
        car_control_msg.omega = u[1]  # omega

        self.pub_car_cmd.publish(car_control_msg)

    @staticmethod
    def trim(value, low, high):
        """
        Trims a value to be between some bounds.

        Args:
            value: the value to be trimmed
            low: the minimum bound
            high: the maximum bound

        Returns:
            the trimmed value
        """
        return max(min(value, high), low)

    def angle_clamp(self, theta):
        if theta > 2 * np.pi:
            return theta - 2 * np.pi
        elif theta < -2 * np.pi:
            return theta + 2 * np.pi
        else:
            return theta

    def read_params_from_calibration_file(self):
        """
        Reads the saved parameters from `/data/config/calibrations/kinematics/DUCKIEBOTNAME.yaml`
        or uses the default values if the file doesn't exist. Adjusts the ROS parameters for the
        node with the new values.
        """

        def readFile(fname):
            with open(fname, "r") as in_file:
                try:
                    return yaml.load(in_file, Loader=yaml.FullLoader)
                except yaml.YAMLError as exc:
                    self.logfatal(
                        "YAML syntax error. File: %s fname. Exc: %s" % (fname, exc)
                    )
                    return None

        # Check file existence
        cali_file_folder = "/data/config/calibrations/kinematics/"
        fname = cali_file_folder + self.veh + ".yaml"
        # Use the default values from the config folder if a robot-specific file does not exist.
        if not os.path.isfile(fname):
            fname = cali_file_folder + "default.yaml"
            self.logwarn(
                "Kinematic calibration %s not found! Using default instead." % fname
            )
            return readFile(fname)
        else:
            return readFile(fname)

    def load_extrinsics(self) -> Union[Homography, None]:
        """
        Loads the homography matrix from the extrinsic calibration file.

        Returns:
            :obj:`Homography`: the loaded homography matrix

        """
        # load extrinsic calibration
        cali_file_folder = "/data/config/calibrations/camera_extrinsic/"
        cali_file = cali_file_folder + rospy.get_namespace().strip("/") + ".yaml"

        # Locate calibration yaml file or use the default otherwise
        if not os.path.isfile(cali_file):
            self.log(
                f"Can't find calibration file: {cali_file}\n Using default calibration instead.",
                "warn",
            )
            cali_file = os.path.join(cali_file_folder, "default.yaml")

        # Shutdown if no calibration file not found
        if not os.path.isfile(cali_file):
            msg = "Found no calibration file ... aborting"
            self.logerr(msg)
            rospy.signal_shutdown(msg)

        try:
            self.H: Homography = HomographyToolkit.load_from_disk(
                cali_file, return_date=False
            )  # type: ignore
            return self.H.reshape((3, 3))
        except Exception as e:
            msg = f"Error in parsing calibration file {cali_file}:\n{e}"
            self.logerr(msg)
            rospy.signal_shutdown(msg)

    def on_shutdown(self):
        self.loginfo("Stopping motors...")
        self.publish_command([0, 0])
        time.sleep(0.5)
        self.loginfo("Motors stopped.")


def rescale(a: float, L: float, U: float):
    if np.allclose(L, U):
        return 0.0
    return (a - L) / (U - L)


if __name__ == "__main__":
    # Initialize the node
    encoder_pose_node = LaneServoingNode(node_name="visual_lane_servoing_node")
    # Keep it spinning
    rospy.spin()

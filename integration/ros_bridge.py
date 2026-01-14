"""
ROS/ROS2 Bridge Module

Provides integration with ROS (Robot Operating System) for:
- Publishing actions to robot
- Subscribing to sensor data
- Service calls for robot control
"""

import torch
import numpy as np
from typing import Optional, Dict, List, Callable, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import threading
import queue
import time


@dataclass
class ROSConfig:
    """Configuration for ROS bridge."""
    # Node settings
    node_name: str = "vla_controller"
    namespace: str = ""

    # Topics
    image_topic: str = "/camera/image_raw"
    depth_topic: str = "/camera/depth"
    lidar_topic: str = "/velodyne_points"
    odom_topic: str = "/odom"
    cmd_vel_topic: str = "/cmd_vel"
    joint_state_topic: str = "/joint_states"
    joint_cmd_topic: str = "/joint_commands"

    # Control
    control_rate: float = 30.0  # Hz
    action_timeout: float = 0.5  # seconds

    # Robot type
    robot_type: str = "manipulator"  # "manipulator", "mobile", "humanoid"


class SensorData:
    """Container for sensor data from ROS."""

    def __init__(self):
        self.image: Optional[np.ndarray] = None
        self.depth: Optional[np.ndarray] = None
        self.lidar: Optional[np.ndarray] = None
        self.joint_positions: Optional[np.ndarray] = None
        self.joint_velocities: Optional[np.ndarray] = None
        self.odom_position: Optional[np.ndarray] = None
        self.odom_velocity: Optional[np.ndarray] = None
        self.timestamp: float = 0.0

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary."""
        return {
            "image": self.image,
            "depth": self.depth,
            "lidar": self.lidar,
            "joint_positions": self.joint_positions,
            "joint_velocities": self.joint_velocities,
            "odom_position": self.odom_position,
            "odom_velocity": self.odom_velocity,
        }


class ROSBridge(ABC):
    """
    Abstract base class for ROS integration.

    Provides interface for VLA model to interact with ROS-based robots.
    """

    def __init__(self, config: ROSConfig):
        self.config = config
        self.is_initialized = False
        self.sensor_data = SensorData()

        # Thread safety
        self.data_lock = threading.Lock()
        self.action_queue = queue.Queue(maxsize=10)

        # Callbacks
        self.sensor_callbacks: List[Callable] = []

    @abstractmethod
    def initialize(self):
        """Initialize ROS node and subscriptions."""
        pass

    @abstractmethod
    def shutdown(self):
        """Shutdown ROS node."""
        pass

    @abstractmethod
    def publish_action(self, action: np.ndarray):
        """Publish action to robot."""
        pass

    @abstractmethod
    def get_sensor_data(self) -> SensorData:
        """Get latest sensor data."""
        pass

    def add_sensor_callback(self, callback: Callable[[SensorData], None]):
        """Add callback for new sensor data."""
        self.sensor_callbacks.append(callback)

    def _notify_callbacks(self, data: SensorData):
        """Notify all registered callbacks."""
        for callback in self.sensor_callbacks:
            try:
                callback(data)
            except Exception as e:
                print(f"Callback error: {e}")


class ROSBridgeImpl(ROSBridge):
    """
    ROS1 implementation of the bridge.

    Requires: rospy, sensor_msgs, geometry_msgs, std_msgs
    """

    def __init__(self, config: ROSConfig):
        super().__init__(config)
        self.subscribers = []
        self.publishers = {}

    def initialize(self):
        """Initialize ROS1 node."""
        try:
            import rospy
            from sensor_msgs.msg import Image, PointCloud2, JointState
            from nav_msgs.msg import Odometry
            from geometry_msgs.msg import Twist
            from std_msgs.msg import Float64MultiArray

            self.rospy = rospy
            self.Image = Image
            self.PointCloud2 = PointCloud2
            self.JointState = JointState
            self.Odometry = Odometry
            self.Twist = Twist
            self.Float64MultiArray = Float64MultiArray

        except ImportError:
            print("ROS1 not available. Using mock implementation.")
            self._use_mock()
            return

        # Initialize node
        rospy.init_node(self.config.node_name, anonymous=True)

        # Setup subscribers
        self._setup_subscribers()

        # Setup publishers
        self._setup_publishers()

        self.is_initialized = True
        print(f"ROS1 bridge initialized: {self.config.node_name}")

    def _setup_subscribers(self):
        """Setup ROS subscribers."""
        # Image subscriber
        sub = self.rospy.Subscriber(
            self.config.image_topic,
            self.Image,
            self._image_callback,
            queue_size=1,
        )
        self.subscribers.append(sub)

        # Joint state subscriber
        sub = self.rospy.Subscriber(
            self.config.joint_state_topic,
            self.JointState,
            self._joint_state_callback,
            queue_size=1,
        )
        self.subscribers.append(sub)

        # Odometry subscriber
        sub = self.rospy.Subscriber(
            self.config.odom_topic,
            self.Odometry,
            self._odom_callback,
            queue_size=1,
        )
        self.subscribers.append(sub)

    def _setup_publishers(self):
        """Setup ROS publishers."""
        if self.config.robot_type == "mobile":
            self.publishers["cmd_vel"] = self.rospy.Publisher(
                self.config.cmd_vel_topic,
                self.Twist,
                queue_size=1,
            )
        else:
            self.publishers["joint_cmd"] = self.rospy.Publisher(
                self.config.joint_cmd_topic,
                self.Float64MultiArray,
                queue_size=1,
            )

    def _image_callback(self, msg):
        """Handle incoming image."""
        try:
            from cv_bridge import CvBridge
            bridge = CvBridge()
            image = bridge.imgmsg_to_cv2(msg, "rgb8")

            with self.data_lock:
                self.sensor_data.image = image
                self.sensor_data.timestamp = msg.header.stamp.to_sec()

            self._notify_callbacks(self.sensor_data)
        except Exception as e:
            print(f"Image callback error: {e}")

    def _joint_state_callback(self, msg):
        """Handle incoming joint state."""
        with self.data_lock:
            self.sensor_data.joint_positions = np.array(msg.position)
            self.sensor_data.joint_velocities = np.array(msg.velocity)

    def _odom_callback(self, msg):
        """Handle incoming odometry."""
        with self.data_lock:
            pos = msg.pose.pose.position
            self.sensor_data.odom_position = np.array([pos.x, pos.y, pos.z])

            vel = msg.twist.twist.linear
            self.sensor_data.odom_velocity = np.array([vel.x, vel.y, vel.z])

    def publish_action(self, action: np.ndarray):
        """Publish action to robot."""
        if not self.is_initialized:
            return

        if self.config.robot_type == "mobile":
            msg = self.Twist()
            msg.linear.x = float(action[0])
            msg.angular.z = float(action[1]) if len(action) > 1 else 0.0
            self.publishers["cmd_vel"].publish(msg)
        else:
            msg = self.Float64MultiArray()
            msg.data = action.tolist()
            self.publishers["joint_cmd"].publish(msg)

    def get_sensor_data(self) -> SensorData:
        """Get latest sensor data."""
        with self.data_lock:
            return self.sensor_data

    def shutdown(self):
        """Shutdown ROS node."""
        if self.is_initialized:
            for sub in self.subscribers:
                sub.unregister()
            self.rospy.signal_shutdown("VLA bridge shutdown")
            self.is_initialized = False

    def _use_mock(self):
        """Use mock implementation when ROS not available."""
        self.is_initialized = True
        print("Using mock ROS implementation")


class ROS2Bridge(ROSBridge):
    """
    ROS2 implementation of the bridge.

    Requires: rclpy, sensor_msgs, geometry_msgs
    """

    def __init__(self, config: ROSConfig):
        super().__init__(config)
        self.node = None
        self.executor = None
        self.spin_thread = None

    def initialize(self):
        """Initialize ROS2 node."""
        try:
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import Image, PointCloud2, JointState
            from nav_msgs.msg import Odometry
            from geometry_msgs.msg import Twist
            from std_msgs.msg import Float64MultiArray

            self.rclpy = rclpy
            self.NodeClass = Node
            self.Image = Image
            self.Twist = Twist
            self.JointState = JointState
            self.Odometry = Odometry
            self.Float64MultiArray = Float64MultiArray

        except ImportError:
            print("ROS2 not available. Using mock implementation.")
            self._use_mock()
            return

        # Initialize ROS2
        rclpy.init()

        # Create node
        self.node = self._create_node()

        # Start spinning in background thread
        self.executor = rclpy.executors.MultiThreadedExecutor()
        self.executor.add_node(self.node)
        self.spin_thread = threading.Thread(target=self._spin, daemon=True)
        self.spin_thread.start()

        self.is_initialized = True
        print(f"ROS2 bridge initialized: {self.config.node_name}")

    def _create_node(self):
        """Create ROS2 node with subscriptions and publishers."""
        class VLANode(self.NodeClass):
            def __init__(node_self, config, bridge):
                super().__init__(config.node_name)
                node_self.bridge = bridge
                node_self.config = config

                # Create subscribers
                node_self.image_sub = node_self.create_subscription(
                    bridge.Image,
                    config.image_topic,
                    node_self.image_callback,
                    10,
                )

                node_self.joint_sub = node_self.create_subscription(
                    bridge.JointState,
                    config.joint_state_topic,
                    node_self.joint_callback,
                    10,
                )

                # Create publishers
                if config.robot_type == "mobile":
                    node_self.cmd_pub = node_self.create_publisher(
                        bridge.Twist,
                        config.cmd_vel_topic,
                        10,
                    )
                else:
                    node_self.cmd_pub = node_self.create_publisher(
                        bridge.Float64MultiArray,
                        config.joint_cmd_topic,
                        10,
                    )

            def image_callback(node_self, msg):
                try:
                    from cv_bridge import CvBridge
                    bridge = CvBridge()
                    image = bridge.imgmsg_to_cv2(msg, "rgb8")

                    with node_self.bridge.data_lock:
                        node_self.bridge.sensor_data.image = image

                    node_self.bridge._notify_callbacks(node_self.bridge.sensor_data)
                except Exception as e:
                    node_self.get_logger().error(f"Image callback error: {e}")

            def joint_callback(node_self, msg):
                with node_self.bridge.data_lock:
                    node_self.bridge.sensor_data.joint_positions = np.array(msg.position)
                    node_self.bridge.sensor_data.joint_velocities = np.array(msg.velocity)

        return VLANode(self.config, self)

    def _spin(self):
        """Spin ROS2 executor."""
        try:
            self.executor.spin()
        except Exception as e:
            print(f"ROS2 spin error: {e}")

    def publish_action(self, action: np.ndarray):
        """Publish action to robot."""
        if not self.is_initialized or self.node is None:
            return

        if self.config.robot_type == "mobile":
            msg = self.Twist()
            msg.linear.x = float(action[0])
            msg.angular.z = float(action[1]) if len(action) > 1 else 0.0
        else:
            msg = self.Float64MultiArray()
            msg.data = action.tolist()

        self.node.cmd_pub.publish(msg)

    def get_sensor_data(self) -> SensorData:
        """Get latest sensor data."""
        with self.data_lock:
            return self.sensor_data

    def shutdown(self):
        """Shutdown ROS2 node."""
        if self.is_initialized:
            if self.executor:
                self.executor.shutdown()
            if self.node:
                self.node.destroy_node()
            self.rclpy.shutdown()
            self.is_initialized = False

    def _use_mock(self):
        """Use mock implementation."""
        self.is_initialized = True


class VLAROSController:
    """
    High-level controller that connects VLA model with ROS.

    Manages the control loop:
    1. Receive sensor data from ROS
    2. Process through VLA model
    3. Publish actions back to robot
    """

    def __init__(
        self,
        model,
        ros_bridge: ROSBridge,
        device: str = "cuda",
    ):
        self.model = model
        self.ros_bridge = ros_bridge
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(self.device)
        self.model.eval()

        self.is_running = False
        self.control_thread = None

    def start(self):
        """Start the control loop."""
        if not self.ros_bridge.is_initialized:
            self.ros_bridge.initialize()

        self.is_running = True
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()

    def stop(self):
        """Stop the control loop."""
        self.is_running = False
        if self.control_thread:
            self.control_thread.join(timeout=2.0)

    def _control_loop(self):
        """Main control loop."""
        rate = 1.0 / self.ros_bridge.config.control_rate

        while self.is_running:
            start_time = time.time()

            try:
                # Get sensor data
                sensor_data = self.ros_bridge.get_sensor_data()

                if sensor_data.image is not None:
                    # Prepare input for model
                    observation = self._prepare_observation(sensor_data)

                    # Get action from model
                    with torch.no_grad():
                        action = self._get_action(observation)

                    # Publish action
                    self.ros_bridge.publish_action(action)

            except Exception as e:
                print(f"Control loop error: {e}")

            # Maintain control rate
            elapsed = time.time() - start_time
            if elapsed < rate:
                time.sleep(rate - elapsed)

    def _prepare_observation(self, sensor_data: SensorData) -> Dict[str, torch.Tensor]:
        """Prepare sensor data for model input."""
        # Process image
        image = sensor_data.image
        if image is not None:
            image = torch.tensor(image, dtype=torch.float32)
            image = image.permute(2, 0, 1) / 255.0  # HWC -> CHW, normalize
            image = image.unsqueeze(0).to(self.device)

        # Create dummy text input (or use actual instruction)
        input_ids = torch.zeros(1, 32, dtype=torch.long, device=self.device)
        attention_mask = torch.ones(1, 32, device=self.device)

        return {
            "pixel_values": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def _get_action(self, observation: Dict[str, torch.Tensor]) -> np.ndarray:
        """Get action from VLA model."""
        outputs = self.model(
            pixel_values=observation["pixel_values"],
            input_ids=observation["input_ids"],
            attention_mask=observation["attention_mask"],
        )

        action = outputs["predicted_actions"]
        return action[0].cpu().numpy()


if __name__ == "__main__":
    # Test ROS bridge creation
    config = ROSConfig(
        node_name="test_vla",
        robot_type="manipulator",
    )

    # Test ROS1 bridge
    ros1_bridge = ROSBridgeImpl(config)
    print("ROS1 bridge created (not initialized)")

    # Test ROS2 bridge
    ros2_bridge = ROS2Bridge(config)
    print("ROS2 bridge created (not initialized)")

    print("\nROS bridges ready for use with VLA models")

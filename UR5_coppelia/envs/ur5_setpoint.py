from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

@dataclass
class PosControl:
    action_dimension: str= 'rad'
    action_low = - 2 * np.pi
    action_high = 2 * np.pi

@dataclass
class VelControl:
    action_dimension: str = 'rad/s'
    action_low = - 0.5 * np.pi
    action_high = 0.5 * np.pi


@dataclass
class ForceControl:
    action_dimension: str = 'N*m'
    action_low = 0.0
    action_high = 200.0

class UR5Env(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 30
    }

    def __init__(self,
                 render_mode=None,
                 scene_path=None,
                 max_episode_steps=1000,
                 control_mode='position',  # 'position', 'velocity', 'force'
                 dynamics_timestep=0.001,
                 simulation_timestep=0.005,
                 enable_collision_check = False,
                 ):
        super().__init__()

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.current_step = 0

        if control_mode not in ['position', 'velocity', 'force']:
            raise ValueError(f"control_mode must be one of ['position', 'velocity', 'force'], got {control_mode}")
        self.control_mode = control_mode

        # 初始化CoppeliaSim连接
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')
        self.sim.stopSimulation()

        self.sim.startSimulation()

        # 设置仿真参数
        self.sim.setFloatProperty(self.sim.handle_scene,'dynamicsStepSize', dynamics_timestep)

        self.sim.setFloatProperty(self.sim.handle_scene, 'timeStep', simulation_timestep)

        self.sim.setStepping(True)

        # 机器人名称
        robot_name = 'UR5'
        gripper_name = 'RG2'
        target_name = 'target'
        tip_name = 'attachPoint'

        # 获取机器人对象
        self.robot = self.sim.getObject(f'/{robot_name}')
        self.gripper = self.sim.getObject(f'/{gripper_name}')
        self.robot_collection = self.sim.createCollection(0)
        self.sim.addItemToCollection(self.robot_collection, self.sim.handle_tree, self.robot, 2)
        self.obstacle_collection = self.sim.createCollection(1)
        self.sim.addItemToCollection(self.obstacle_collection, self.sim.handle_tree, self.sim.getObject('/obstacle'), 2)
        self.target = self.sim.getObject(f'/{target_name}')
        self.tip = self.sim.getObject(f'/{tip_name}')

        # 获取关节
        self.robot_joints = self.sim.getObjectsInTree(self.robot, self.sim.sceneobject_joint, 1)
        self.gripper_joints = self.sim.getObjectsInTree(self.gripper, self.sim.sceneobject_joint, 1)
        self.general_joints = [joint for joint in self.robot_joints if joint not in self.gripper_joints]

        # 确保有6个关节
        assert len(self.general_joints) == 6, f"Expected 6 joints, got {len(self.general_joints)}"

        self.pos_intervals = [self.sim.getFloatArrayProperty(joint, 'interval') for joint in self.general_joints]
        self.maxVelAccJerk = [self.sim.getFloatArrayProperty(joint, 'maxVelAccelJerk') for joint in self.general_joints]

        # if assign
        # check availability
        if control_mode == 'position':
            self.action_low, self.action_high = PosControl.action_low, PosControl.action_high
        elif control_mode == 'velocity':
            self.action_low, self.action_high = VelControl.action_low, VelControl.action_high
        elif control_mode == 'force':
            self.action_low, self.action_high = ForceControl.action_low, ForceControl.action_high


        # 获取手眼摄像头
        self.eye_in_hand_sensor = self.sim.getObject('./eyehandsensor')
        self.fixed_sensor = self.sim.getObject('./fixedsensor/rgb')

        # 定义动作空间：根据控制模式调整
        self.action_space = spaces.Box(
            low=np.ones(shape=(6,),dtype=np.float32)*self.action_low, high=np.ones(shape=(6,),dtype=np.float32)*self.action_high, shape=(6,), dtype=np.float32
        )

        # 定义观测空间
        # 关节状态：6个关节的位置和速度
        joint_obs_shape = (12,)  # 6个位置 + 6个速度

        # 图像观测：假设摄像头分辨率为64x64 RGB
        img_shape = (64, 64, 3)

        # 组合观测空间
        # self.observation_space = spaces.Dict({
        #     'joints': spaces.Box(low=-np.inf, high=np.inf, shape=joint_obs_shape, dtype=np.float32),
        #     'eye_in_hand_image': spaces.Box(low=0, high=255, shape=img_shape, dtype=np.uint8),
        #     'fixed_image': spaces.Box(low=0, high=255, shape=img_shape, dtype=np.uint8)
        # })
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=joint_obs_shape, dtype=np.float32)

        self.enable_collision_check = enable_collision_check
        # 初始化状态
        self.state = None
        self.collision = None

    def reset(self, seed=None, options=None):
        """
        重置环境
        """

        super().reset(seed=seed)
        self.current_step = 0

        # 随机初始化关节位置
        initial_positions = self.np_random.uniform(-np.pi / 3, np.pi / 3, size=6)

        # 设置关节为被动模式进行重置
        self._set_joint_mode('kinematic')
        for i, joint in enumerate(self.general_joints):
            self.sim.setJointPosition(joint, float(initial_positions[i]))

        self.sim.step()

        # 获取初始观测
        observation = self._get_observation()
        self.state = observation

        # 获取初始信息
        info = self._get_info()

        # 设置关节为指定的控制模式
        self._set_joint_mode(self.control_mode)

        return observation, info

    def step(self, action):
        """
        执行一步动作
        """
        for i, joint in enumerate(self.general_joints):
            act = action[i]
            if self.control_mode == 'position':
                self.sim.setJointTargetPosition(joint, float(act))
            elif self.control_mode == 'velocity':
                self.sim.setJointTargetVelocity(joint, float(act))
            elif self.control_mode == 'force':
                self.sim.setJointTargetForce(joint, float(act))

        self.sim.step()

        # 增加步数计数器
        self.current_step += 1

        # 获取新观测
        observation = self._get_observation()
        self.state = observation

        # 计算奖励
        reward = 0.0

        # 检查是否终止
        terminated = False

        result, self.collision = self.sim.checkCollision(self.robot_collection, self.obstacle_collection)

        tip_pose = np.asarray(self.sim.getObjectPose(self.tip)[:3])
        target_pose = np.asarray(self.sim.getObjectPose(self.target)[:3])

        if np.mean(np.square(tip_pose-target_pose))<1e-3:
            terminated = True
            reward += 50.0

        reward -= 10.0 if bool(result) else 0.1

        terminated = (bool(result) or terminated) if self.enable_collision_check else terminated

        # 检查是否截断（超过最大步数）
        truncated = self.current_step >= self.max_episode_steps

        # 获取信息
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _set_joint_mode(self, mode: str):
        if mode == 'force':
            # 设置关节模式为动态
            for joint in self.general_joints:
                self.sim.setJointMode(joint, self.sim.jointmode_dynamic)
                self.sim.setIntProperty(joint, 'dynCtrlMode', self.sim.jointdynctrl_force)
        elif mode == 'position':
            for joint in self.general_joints:
                self.sim.setJointMode(joint, self.sim.jointmode_dynamic)
                self.sim.setIntProperty(joint, 'dynCtrlMode', self.sim.jointdynctrl_position)
        elif mode == 'kinematic':
            for joint in self.general_joints:
                self.sim.setJointMode(joint, self.sim.jointmode_kinematic)
        elif mode == 'velocity':
            for joint in self.general_joints:
                self.sim.setJointMode(joint, self.sim.jointmode_dynamic)
                self.sim.setIntProperty(joint, 'dynCtrlMode', self.sim.jointdynctrl_velocity)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _get_observation(self):
        """
        获取当前观测
        """
        # 获取关节状态
        joint_positions = np.array([self.sim.getJointPosition(joint) for joint in self.general_joints])
        joint_velocities = np.array([self.sim.getJointVelocity(joint) for joint in self.general_joints])
        joint_states = np.concatenate([joint_positions, joint_velocities])

        # 获取手眼摄像头图像
        eye_in_hand_img = self._get_camera_image(self.eye_in_hand_sensor)
        fixed_img = self._get_camera_image(self.fixed_sensor)

        # observation = {
        #     'joints': joint_states.astype(np.float32),
        #     'eye_in_hand_image': eye_in_hand_img,
        #     'fixed_image': fixed_img
        # }
        observation = joint_states.astype(np.float32)

        return observation

    def _get_camera_image(self, sensor_handle):
        """
        获取摄像头图像并调整为64x64
        """
        # 获取图像数据
        options = 0  # RGB图像
        data, resolution = self.sim.getVisionSensorImg(sensor_handle, options)
        data = self.sim.unpackUInt8Table(data)

        # 转换为numpy数组
        img = np.asarray(data, dtype=np.uint8).reshape(*resolution, 3)

        # 调整图像大小为64x64
        # 注意：这里需要安装opencv-python
        try:
            import cv2
            img = cv2.resize(img, (64, 64))
        except ImportError:
            h, w = img.shape[:2]
            if h != 64 or w != 64:
                # 简单裁剪或填充
                if h > 64:
                    img = img[:64, :]
                if w > 64:
                    img = img[:, :64]
                if h < 64 or w < 64:
                    padded_img = np.zeros((64, 64, 3), dtype=np.uint8)
                    padded_img[:h, :w] = img
                    img = padded_img

        return img

    def _get_info(self):
        """
        获取额外信息
        """
        return {
            'step': self.current_step,
            'current_time': self.sim.getSimulationTime(),
            'target': [self.sim.getJointTargetPosition(joint) for joint in self.general_joints],
            'joint_positions': [self.sim.getJointPosition(joint) for joint in self.general_joints],
            'collision pairs': self.collision
        }

    def render(self):
        pass

    def close(self):
        """
        关闭环境
        """
        if hasattr(self, 'sim'):
            self.sim.stopSimulation()
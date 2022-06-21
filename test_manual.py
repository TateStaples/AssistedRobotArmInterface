from numpy.linalg import inv
from interface_supervisory import KinectView as Supervisory
from interface_supervisory import *
import threading

debugging = False
camera = False
robot = False
mac = True
logging = True


class KinectView(Supervisory):
    main_flag = True

    # movement - None = still
    running = True

    # ---------- util functions ---------- #

    def make_button(self, press, text="", pos=(0, 0), size=None, font_size=20, background_normal=None, border=None,
                    release=None):
        if size is None: size = self.button_size_1
        pos = self.pos_hint_from_tuple(pos)
        if background_normal is None:  # if labeled box
            b = Button(text=text, pos_hint=pos, font_name=self.text_font, font_size=font_size, size=size,
                       size_hint=(None, None))
            b.background_color = self.button_color + (1,)
        else:  # if image
            b = Button(pos_hint=pos, size=size, size_hint=(None, None), background_normal=background_normal,
                       border=border)
        b.bind(on_press=self.function_wrapper(press))
        if release is not None:
            b.bind(on_release=self.function_wrapper(release))
        self.root.add_widget(b)
        return b

    def make_label(self, text, pos):
        font = self.title_font
        pos = self.pos_hint_from_tuple(pos)
        label = Label(text=text, pos_hint=pos, font_name=self.text_font, font_size=font, size_hint=(None, None),
                      halign="left",
                      color=self.text_color + (1,))
        label.bind(texture_size=label.setter('size'))
        self.root.add_widget(label)
        return label

    # ---------- init ---------- #
    def build(self):
        self.start_log()
        self.root = layout = FloatLayout()

        self.build_sizes()

        # ----- session control ----- #
        self.build_session_control()

        # ----- camera ----- #
        self.build_cams()

        # ----- nudge controls ----- #
        self.build_move()
        self.build_gripper()

        with layout.canvas.before:
            Color(self.background_color[0], self.background_color[1], self.background_color[2], 1)
            self.rect = Rectangle(
                size=(int(Config.get('graphics', 'width')) * 2, int(Config.get('graphics', 'height')) * 2),
                pos=layout.pos)

        Thread(target=self.move_thread).start()
        Thread(target=self.rotate_thread).start()
        # Thread(target=self.update_img).start()

        return layout

    def build_sizes(self):
        w, h = int(Config.get('graphics', 'width')), int(Config.get('graphics', 'height'))
        if not mac:
            w, h = w // 2, h // 2

        self.title_font = int(w / 15)

        img_size = i_w, i_h = (640, 480)
        self.main_img_size = img_size
        scale_factor = 1 / 2
        self.secondary_img_size = (int(i_w * scale_factor), int(i_h * scale_factor))

        self.button_size_1 = 1 / 6 * w, 1 / 6 * h
        self.return_button_size = self.button_size_1[0] * 2.25, self.button_size_1[1]
        self.session_button_size = self.button_size_1
        self.camera_button_size = self.button_size_1[0] * 1.5, self.button_size_1[1]
        self.elevation_button_size = self.button_size_1[0], self.button_size_1[1]
        self.gripper_button_size = self.elevation_button_size

        self.rotation_button_size = 1 / 6 * w, 1 / 4 * h
        self.left_right_size = 1 / 8 * w, 1 / 10 * h
        self.forward_backward_size = 1 / 10 * w, 1 / 10 * h

    def build_session_control(self):
        s_x, s_y = 0.05, 0.15
        self.session_label = self.make_label("Session Control", (s_x, s_y))
        self.session_start = self.make_button(self.on_session_start, text="Start", size=self.session_button_size,
                                              pos=(s_x, s_y - .1))
        self.session_stop = self.make_button(self.on_stop, text="Stop", size=self.session_button_size,
                                             pos=(s_x + .1, s_y - .1))

    def build_cams(self):
        self.establish_cameras()
        c_x, c_y = 0.2, 0.3
        self.switchCamButton = self.make_button(self.switch_camera, text="Switch View", pos=(c_x - .005, c_y - .03),
                                                size=self.camera_button_size)
        pos_main = self.pos_hint_from_tuple((0.05, c_y + 0.075))
        w, h = self.dimensions()
        i_w, i_h = self.main_img_size
        hint_w, hint_h = i_w / w, i_h / h
        left_hint = pos_main['x'] + hint_w
        pos_secondary = self.pos_hint_from_tuple((left_hint - (self.secondary_img_size[0] / w), 0.05))
        self.main_img = Image(on_touch_down=self.on_press_image, pos_hint=pos_main, size=self.main_img_size,
                              source="resource_manual/main.png", size_hint=(None, None))
        self.secondary_img = Image(pos_hint=pos_secondary, size=self.secondary_img_size,
                                   source="resource_manual/secondary.png", size_hint=(None, None))
        # self.root.bind(size=self._update_rect, pos=self._update_rect)
        self.root.add_widget(self.main_img)
        self.root.add_widget(self.secondary_img)
        Clock.schedule_interval(self.update_img, 1 / 60)

    def build_move(self):
        w, h = self.dimensions()
        border = (0, 0, 0, 0)
        far_left = self.main_img.pos_hint['x'] + self.main_img.size[0] / w
        m_x, m_y = far_left + 0.05, 0.4
        elevation_y = m_y + .1
        move_offset = .02

        self.armLabel = self.make_label("Arm Movement", (m_x, m_y + .51))
        self.returnButton = self.make_button(self.on_press_return, size=self.return_button_size,
                                             text="Go to start position", pos=(m_x + .02, m_y - .392 + .04))
        self.upButton = self.make_button(self.on_press_upButton, text="Lift", release=self.on_release_upBotton,
                                         pos=(m_x + move_offset, elevation_y + .06), size=self.elevation_button_size)
        self.downButton = self.make_button(self.on_press_downButton, release=self.on_release_downButton,
                                           text="Lower", pos=(m_x + move_offset + .1, elevation_y + .06),
                                           size=self.elevation_button_size)

        w1, h1 = self.left_right_size
        w2, h2 = self.forward_backward_size
        w, h = self.dimensions()
        d_long_x, d_short_x = w1 / w, w2 / w
        d_long_y, d_short_y = h2 / h, h1 / h

        left_right_y = m_y + .3
        forward_backward_x = m_x + .08 - d_short_x / 4

        self.leftButton = self.make_button(self.on_press_leftButton,
                                           background_normal="resources/left.png", size=self.left_right_size,
                                           pos=(forward_backward_x + move_offset - d_long_x, left_right_y + .06),
                                           border=border, release=self.on_release_leftButton)
        self.rightButton = self.make_button(self.on_press_rightButton,
                                            background_normal="resources/right.png", size=self.left_right_size,
                                            pos=(forward_backward_x + move_offset + d_short_x, left_right_y + .06),
                                            border=border, release=self.on_release_rightButton)
        self.backwardButton = self.make_button(self.on_press_backwardButton,
                                               background_normal="resources/backward.png",
                                               size=self.forward_backward_size,
                                               pos=(forward_backward_x + move_offset, left_right_y - d_long_y + .06),
                                               border=border, release=self.on_release_backwardButton)
        self.forwardButton = self.make_button(self.on_press_forwardButton,
                                              background_normal="resources/forward.png",
                                              size=self.forward_backward_size,
                                              pos=(forward_backward_x + move_offset, left_right_y + d_short_y + .06),
                                              border=border, release=self.on_release_forwardButton)

    def build_gripper(self):
        g_x, g_y = self.armLabel.pos_hint['x'], 0
        open_close_y = 0.13
        rotate_xoffset = .02
        gripper_button_size = self.button_size_1

        self.gripperLabel = self.make_label("Gripper Rotation", (g_x, g_y + 0.4 + .06))

        self.gripperButtonOpen1 = self.make_button(self.on_press_gripperButtonOpen, text="Open \nGripper",
                                                   size=gripper_button_size,
                                                   pos=(g_x + rotate_xoffset, open_close_y + .06))

        self.gripperButtonClose1 = self.make_button(self.on_press_gripperButtonClose, text="Close \nGripper",
                                                    size=gripper_button_size,
                                                    pos=(g_x + rotate_xoffset + .1, open_close_y + .06))

        width, _ = self.dimensions()
        far_right = self.gripperButtonClose1.size[0] / width + self.gripperButtonClose1.pos_hint['x']
        center_x = (g_x + far_right) / 2
        button_width = self.rotation_button_size[0] / width
        offset = 0.005

        rotate_y = 0.25
        border = (0, 0, 0, 0)
        self.rotateCounterClock = self.make_button(self.on_press_rotate_counter_clock,
                                                   release=self.on_release_rotate_counter_clock,
                                                   background_normal="resources/counter_clock.png",
                                                   size=self.rotation_button_size,
                                                   pos=(
                                                       center_x - button_width - offset + rotate_xoffset,
                                                       rotate_y + .06),
                                                   border=border)
        self.rotateClock = self.make_button(self.on_press_rotate_clock, release=self.on_release_rotate_clock,
                                            background_normal="resources/clock.png", size=self.rotation_button_size,
                                            pos=(center_x + offset + rotate_xoffset, rotate_y + .06), border=border)

    # ---------- main ---------- #
    def establish_cameras(self, new=True):
        if camera:
            if new:
                self.pipeline1 = rs.pipeline()
                self.pipeline2 = rs.pipeline()
            profile1 = self.pipeline1.start(  # outer camera
                create_config(device_1, self.main_img_size if self.main_flag else self.secondary_img_size,
                              self.main_flag))
            profile2 = self.pipeline2.start(  # arm camera
                create_config(device_2, self.main_img_size if not self.main_flag else self.secondary_img_size,
                              not self.main_flag))
            self.depth_scale = (
                profile1 if self.main_flag else profile2).get_device().first_depth_sensor().get_depth_scale()

    # def switch_camera(self):
    #     self.logDebug("Swapping cameras")
    #     self.back_main_flag = not self.back_main_flag
    #     if camera:
    #         self.pipeline1.stop()
    #         self.pipeline2.stop()
    #     self.establish_cameras(new=False)
    #     self.main_img.reload()
    #     self.secondary_img.reload()
    #     self.delete_flag = True
    def switch_camera(self):
        self.logDebug("Swapping cameras")
        self.main_flag = not self.main_flag
        if camera:
            self.pipeline1.stop()
            self.pipeline2.stop()
        self.establish_cameras(new=False)

    # ---------- movement ---------- #

    def on_press_upButton(self):
        self.upButton.disabled = True
        self.border_color = None
        Thread(target=self.move_thread, args=["up"]).start()
        time.sleep(0.1)

    def on_release_upBotton(self):
        if robot: rob.stopl()
        try:
            self.timer_close.cancel()
            self.timer_limit.cancel()
        except:
            pass
        self.border_color = None
        self.upButton.disabled = False
        Thread(target=self.move_thread, args=["up"])._stop

        # Thread(target=self.pause_thread).start()

    def on_press_downButton(self):
        self.downButton.disabled = True
        self.border_color = None
        Thread(target=self.move_thread, args=["down"]).start()
        time.sleep(0.1)

    def on_release_downButton(self):
        if robot: rob.stopl()
        try:
            self.timer_close.cancel()
            self.timer_limit.cancel()
        except:
            pass
        self.border_color = None
        self.downButton.disabled = False
        Thread(target=self.move_thread, args=["down"])._stop

        # Thread(target=self.pause_thread).start()

    def on_press_leftButton(self):
        self.leftButton.disabled = True
        self.border_color = None
        Thread(target=self.move_thread, args=["left"]).start()
        time.sleep(0.1)

    def on_release_leftButton(self):
        if robot: rob.stopl()
        try:
            self.timer_close.cancel()
            self.timer_limit.cancel()
        except:
            pass
        self.border_color = None
        self.leftButton.disabled = False
        Thread(target=self.move_thread, args=["left"])._stop

        # Thread(target=self.pause_thread).start()

    def on_press_rightButton(self):
        self.rightButton.disabled = True
        self.border_color = None
        Thread(target=self.move_thread, args=["right"]).start()
        time.sleep(0.1)

    def on_release_rightButton(self):
        if robot: rob.stopl()
        try:
            self.timer_close.cancel()
            self.timer_limit.cancel()
        except:
            pass
        self.border_color = None
        self.rightButton.disabled = False
        Thread(target=self.move_thread, args=["right"])._stop
        # Thread(target=self.pause_thread).start()

    def on_press_backwardButton(self):
        self.backwardButton.disabled = True
        self.border_color = None
        Thread(target=self.move_thread, args=["backward"]).start()
        time.sleep(0.1)

    def on_release_backwardButton(self):
        if robot: rob.stopl()
        try:
            self.timer_close.cancel()
            self.timer_limit.cancel()
        except:
            pass
        self.border_color = None
        self.backwardButton.disabled = False
        Thread(target=self.move_thread, args=["backward"])._stop

        # Thread(target=self.pause_thread).start()

    def on_press_forwardButton(self):
        self.forwardButton.disabled = True
        self.border_color = None
        Thread(target=self.move_thread, args=["forward"]).start()
        time.sleep(0.1)

    def on_release_forwardButton(self):
        if robot: rob.stopl()
        try:
            self.timer_close.cancel()
            self.timer_limit.cancel()
        except:
            pass
        self.border_color = None
        self.forwardButton.disabled = False
        Thread(target=self.move_thread, args=["forward"])._stop

        # Thread(target=self.pause_thread).start()

    # ---------- rotation ---------- #
    def on_press_rotate_clock(self):
        self.rotateClock.disabled = True
        self.border_color = None
        Thread(target=self.rotate_thread, args=["rz-"]).start()
        time.sleep(0.1)

    def on_release_rotate_clock(self):
        if robot: rob.stopl()
        try:
            self.timer_close.cancel()
            self.timer_limit.cancel()
        except:
            pass
        self.border_color = None
        self.rotateClock.disabled = False
        Thread(target=self.rotate_thread, args=["rz-"])._stop

        # Thread(target=self.pause_thread).start()

    def on_press_rotate_counter_clock(self):
        self.rotateCounterClock.disabled = True
        self.border_color = None
        Thread(target=self.rotate_thread, args=["rz+"]).start()
        time.sleep(0.1)

    def on_release_rotate_counter_clock(self):
        if robot: rob.stopl()
        self.border_color = None
        try:
            self.timer_close.cancel()
            self.timer_limit.cancel()
        except:
            pass
        self.rotateCounterClock.disabled = False
        Thread(target=self.rotate_thread, args=["rz+"])._stop

    # ---------------------------------------------------------------------------------#

    def move_thread(self, direction):  # this probably needs updating
        if not robot:
            return
        # rob_x, rob_y, rob_z, rob_rx, rob_ry, rob_rz = rob.getl()
        # rob_pos = np.array([rob_x, rob_y, rob_z])
        # camera_pos = np.dot(inversed_R_back, rob_pos.T).T + inversed_t_back
        # cam_x, cam_y, cam_z = camera_pos

        Transform_EE2BASE_move = rob.get_pose()

        T_EE2BASE_move = Transform_EE2BASE_move.array
        R_EE2BASE_move = np.array([[T_EE2BASE_move[0, 0], T_EE2BASE_move[0, 1], T_EE2BASE_move[0, 2]],
                                   [T_EE2BASE_move[1, 0], T_EE2BASE_move[1, 1], T_EE2BASE_move[1, 2]],
                                   [T_EE2BASE_move[2, 0], T_EE2BASE_move[2, 1], T_EE2BASE_move[2, 2]]])
        # translation
        Tr_EE2BASE_move = np.array([T_EE2BASE_move[0, 3], T_EE2BASE_move[1, 3], T_EE2BASE_move[2, 3]])

        T_BASE2EE_move = inv(T_EE2BASE_move)
        # rotation
        R_BASE2EE_move = np.array([[T_BASE2EE_move[0, 0], T_BASE2EE_move[0, 1], T_BASE2EE_move[0, 2]],
                                   [T_BASE2EE_move[1, 0], T_BASE2EE_move[1, 1], T_BASE2EE_move[1, 2]],
                                   [T_BASE2EE_move[2, 0], T_BASE2EE_move[2, 1], T_BASE2EE_move[2, 2]]])
        # translation
        Tr_BASE2EE_move = np.array([T_BASE2EE_move[0, 3], T_BASE2EE_move[1, 3], T_BASE2EE_move[2, 3]])

        rob_x, rob_y, rob_z, rob_rx, rob_ry, rob_rz = rob.getl()
        rob_pos_in_base = np.array([rob_x, rob_y, rob_z])
        rob_pos_in_EE_urx = np.dot(R_BASE2EE_move, rob_pos_in_base.T).T + Tr_BASE2EE_move
        rob_pos_EE_k = np.dot(R_k2ur5_inv, rob_pos_in_EE_urx.T).T
        rob_pos_in_cam = np.dot(R_on_arm_inv, rob_pos_EE_k.T).T + t_on_arm_inv

        cam_x, cam_y, cam_z = rob_pos_in_cam

        MOVE_UNIT = 100
        dist_limit = .85
        up_limit = 0.4
        down_limit = .15
        back_limit = .3

        if direction == "up":
            MOVE_UNIT = up_limit - rob_z
            camera_pos = np.array([cam_x, cam_y, cam_z - MOVE_UNIT])

        if direction == "down":
            MOVE_UNIT = rob_z + down_limit
            camera_pos = np.array([cam_x, cam_y, cam_z + MOVE_UNIT])

        if direction == "left":
            camera_pos = np.array([cam_x - MOVE_UNIT, cam_y, cam_z])

            rob_target_EE_K = np.dot(R_on_arm, camera_pos.T).T + t_on_arm
            rob_target_EE_urx = np.dot(R_k2ur5, rob_target_EE_K.T).T
            rob_target_base = np.dot(R_EE2BASE_move, rob_target_EE_urx.T).T + Tr_EE2BASE_move
            r_x, r_y, r_z = rob_target_base[0], rob_target_base[1], rob_target_base[2]
            target_dir = np.array([r_x - rob_x, r_y - rob_y, r_z - rob_z])

            initial_dist = (target_dir[0] ** 2 + target_dir[1] ** 2 + target_dir[2] ** 2) ** .5
            print("initial_dist = ", initial_dist)
            theta = np.lib.scimath.arccos(
                np.dot(target_dir, rob_target_base) / (np.linalg.norm(target_dir) * np.linalg.norm(rob_target_base)))
            print("theta=", theta)
            print("cos(theta)_true",
                  np.dot(target_dir, rob_target_base) / (np.linalg.norm(target_dir) * np.linalg.norm(rob_target_base)))
            print("cos(theta)", np.cos(theta))

            root_x = np.linalg.norm(rob_target_base)
            delta = ((2 * root_x * np.cos(theta)) ** 2 - 4 * (root_x ** 2 - dist_limit ** 2)) ** .5  # sqrt(b**2-4ac)
            root_b = 2 * root_x * np.cos(theta)

            root_1 = (root_b + delta) * .5
            root_2 = (root_b - delta) * .5
            subtracts = np.sort([root_1, root_2])
            print("subtracts=", subtracts)
            subtract = subtracts[0]
            print("subtract=", subtract)
            MOVE_UNIT = initial_dist - subtract
            if MOVE_UNIT < 0:
                MOVE_UNIT = 0

            print("move unit=", MOVE_UNIT)

            camera_pos = np.array([cam_x - MOVE_UNIT, cam_y, cam_z])

        if direction == "right":
            camera_pos = np.array([cam_x + MOVE_UNIT, cam_y, cam_z])

            rob_target_EE_K = np.dot(R_on_arm, camera_pos.T).T + t_on_arm
            rob_target_EE_urx = np.dot(R_k2ur5, rob_target_EE_K.T).T
            rob_target_base = np.dot(R_EE2BASE_move, rob_target_EE_urx.T).T + Tr_EE2BASE_move
            r_x, r_y, r_z = rob_target_base[0], rob_target_base[1], rob_target_base[2]
            target_dir = np.array([r_x - rob_x, r_y - rob_y, r_z - rob_z])

            initial_dist = (target_dir[0] ** 2 + target_dir[1] ** 2 + target_dir[2] ** 2) ** .5
            print("initial_dist = ", initial_dist)
            theta = np.lib.scimath.arccos(
                np.dot(target_dir, rob_target_base) / (np.linalg.norm(target_dir) * np.linalg.norm(rob_target_base)))
            print("theta=", theta)
            print("cos(theta)_true",
                  np.dot(target_dir, rob_target_base) / (np.linalg.norm(target_dir) * np.linalg.norm(rob_target_base)))
            print("cos(theta)", np.cos(theta))

            root_x = np.linalg.norm(rob_target_base)
            delta = ((2 * root_x * np.cos(theta)) ** 2 - 4 * (root_x ** 2 - dist_limit ** 2)) ** .5  # sqrt(b**2-4ac)
            root_b = 2 * root_x * np.cos(theta)

            root_1 = (root_b + delta) * .5
            root_2 = (root_b - delta) * .5
            subtracts = np.sort([root_1, root_2])
            print("subtracts=", subtracts)
            subtract = subtracts[0]
            print("subtract=", subtract)
            MOVE_UNIT = initial_dist - subtract
            if MOVE_UNIT < 0:
                MOVE_UNIT = 0

            print("move unit=", MOVE_UNIT)

            camera_pos = np.array([cam_x + MOVE_UNIT, cam_y, cam_z])

        if direction == "forward":
            camera_pos = np.array([cam_x, cam_y - MOVE_UNIT, cam_z])

            rob_target_EE_K = np.dot(R_on_arm, camera_pos.T).T + t_on_arm
            rob_target_EE_urx = np.dot(R_k2ur5, rob_target_EE_K.T).T
            rob_target_base = np.dot(R_EE2BASE_move, rob_target_EE_urx.T).T + Tr_EE2BASE_move
            r_x, r_y, r_z = rob_target_base[0], rob_target_base[1], rob_target_base[2]
            target_dir = np.array([r_x - rob_x, r_y - rob_y, r_z - rob_z])

            initial_dist = (target_dir[0] ** 2 + target_dir[1] ** 2 + target_dir[2] ** 2) ** .5
            print("initial_dist = ", initial_dist)
            theta = np.lib.scimath.arccos(
                np.dot(target_dir, rob_target_base) / (np.linalg.norm(target_dir) * np.linalg.norm(rob_target_base)))
            print("theta=", theta)
            print("cos(theta)_true",
                  np.dot(target_dir, rob_target_base) / (np.linalg.norm(target_dir) * np.linalg.norm(rob_target_base)))
            print("cos(theta)", np.cos(theta))

            root_x = np.linalg.norm(rob_target_base)
            delta = ((2 * root_x * np.cos(theta)) ** 2 - 4 * (root_x ** 2 - dist_limit ** 2)) ** .5  # sqrt(b**2-4ac)
            root_b = 2 * root_x * np.cos(theta)

            root_1 = (root_b + delta) * .5
            root_2 = (root_b - delta) * .5
            subtracts = np.sort([root_1, root_2])
            print("subtracts=", subtracts)
            subtract = subtracts[0]
            print("subtract=", subtract)
            MOVE_UNIT = initial_dist - subtract
            if MOVE_UNIT < 0:
                MOVE_UNIT = 0

            print("move unit=", MOVE_UNIT)

            camera_pos = np.array([cam_x, cam_y - MOVE_UNIT, cam_z])

        if direction == "backward":
            # print("dist to base = ",(rob_x**2+rob_y**2)**.5)
            MOVE_UNIT = (rob_x ** 2 + rob_y ** 2) ** .5 - back_limit
            if MOVE_UNIT < 0:
                MOVE_UNIT = 0
            camera_pos = np.array([cam_x, cam_y + MOVE_UNIT, cam_z])

        # rob_pos = np.dot(R_back, camera_pos.T).T + t_back
        rob_target_EE_K = np.dot(R_on_arm, camera_pos.T).T + t_on_arm
        rob_target_EE_urx = np.dot(R_k2ur5, rob_target_EE_K.T).T
        rob_target_base = np.dot(R_EE2BASE_move, rob_target_EE_urx.T).T + Tr_EE2BASE_move
        r_x, r_y, r_z = rob_target_base[0], rob_target_base[1], rob_target_base[2]
        rob_rx, rob_ry, rob_rz = rob.getl()[3:6]

        move_time = MOVE_UNIT / v

        self.logDebug(rob_target_base)
        if MOVE_UNIT > .002:
            self.timer_close = threading.Timer(move_time + 3, self.nudge_close)
            self.timer_limit = threading.Timer(move_time + 5.5, self.nudge_error)
            self.timer_close.start()
            self.timer_limit.start()
        else:
            self.nudge_error()
            time.sleep(1)
        rob.movel((r_x, r_y, r_z, rob_rx, rob_ry, rob_rz), a, v)
        self.border_color = None

    def rotate_thread(self, direction):
        rob_o = rob.get_orientation()
        rob_pos = rob.getl()
        rob_rx, rob_ry, rob_rz = rob_pos[3], rob_pos[4], rob_pos[5]

        if direction == "rz+":
            rob_o.rotate_zb(rob_rz + ROTATE_UNIT)
            rob.set_orientation(rob_o, acc=0.03, vel=0.05)

        elif direction == "rz-":
            rob_o.rotate_zb(rob_rz - ROTATE_UNIT)
            rob.set_orientation(rob_o, acc=0.03, vel=0.05)

    def update_img(self, *args):
        # kivy's recommendation for video streaming:
        # https://kivyapps.wordpress.com/video-streaming-using-kivy-and-python/
        if not camera:
            ret, outer = video.read()
            arm = np.ones(self.secondary_img_size + (3,)) * 150
        else:
            # Get frameset of color and depth
            frames1 = self.pipeline1.wait_for_frames()
            frames2 = self.pipeline2.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames1 = align.process(frames1)
            aligned_frames2 = align.process(frames2)

            # Get aligned frames
            color_frame1 = aligned_frames1.get_color_frame()
            color_frame2 = aligned_frames2.get_color_frame()

            color_image1 = np.asanyarray(color_frame1.get_data())
            color_image2 = np.asanyarray(color_frame2.get_data())

            aligned_depth_frame = aligned_frames1.get_depth_frame() if self.main_flag else aligned_frames2.get_depth_frame()
            self.aligned_depth_intrinsics = rs.video_stream_profile(aligned_depth_frame.profile).get_intrinsics()
            self.depth_image = np.asanyarray(aligned_depth_frame.get_data())

            outer = color_image1
            arm = color_image2
            # cv2.imshow("image 1", color_image1)

        arm = self.camera_mask(arm)

        frame_big = outer if self.main_flag else arm  # cv2.resize(outer if self.main_flag else arm, self.main_img_size)
        frame_small = outer if not self.main_flag else arm  # cv2.resize(outer if not self.main_flag else arm, self.secondary_img_size)
        frame_big = self.draw_notices(frame_big)

        cv2.imwrite("resource_manual/main.png", frame_big)
        cv2.imwrite("resource_manual/secondary.png", frame_small)
        self.main_img.reload()
        self.secondary_img.reload()


MOVE_UNIT = 100
ROTATE_UNIT = 3

Config.set('graphics', 'width', '500' if mac else '1000')
Config.set('graphics', 'height', '400' if mac else '800')

if __name__ == '__main__':
    app = KinectView()
    app.run()
    app.end_log()
    cv2.destroyAllWindows()
import kivy
from kivy import Config
from kivy.app import App
from kivy.uix.gridlayout import GridLayout


class MyApp(App):
    def build(self):
        return MyLayout()


class MyLayout(GridLayout):
    def thing(self, section, module, pnum, experiment, interfacetype):
        section = section.text
        module = module.text
        pnum = int(pnum.value)
        ex = experiment.active
        interface = interfacetype.text
        login.stop()

        file_name = "Section" + section + module + "_" + str(pnum) if ex else None
        if "supervisory" in interface.lower():
            import interface_supervisory as interface
        else:
            import test_manual as interface
        interface.logging = ex
        app = interface.KinectView()
        app.log_file = file_name + ".xlsx" if ex else None
        app.run()
        app.end_log()


if __name__ == '__main__':
    Config.set('graphics', 'width', '1280')
    Config.set('graphics', 'height', '840')
    login = MyApp()
    login.run()


from PyQt5.QtWidgets import QMainWindow, QMessageBox
from PyQt5.uic import loadUi
from settings_window import SettingsWindow
import webbrowser
import requests
import json

class LoginWindow(QMainWindow):
    def __init__(self):
        super(LoginWindow, self).__init__()
        loadUi('login_window.ui', self)

        self.register_button.clicked.connect(self.go_to_register_page)
        self.login_button.clicked.connect(self.login)

        self.popup = QMessageBox()
        self.popup.setWindowTitle("Failed")
        self.show()

    def go_to_register_page(self):
        webbrowser.open('http://127.0.0.1:8000/register/')

    def login(self):
        try:
            url = 'http://127.0.0.1:8000/api/get_auth_token/'
            response = requests.post(url, data={'username': self.username_input.text(), 'password': self.password_input.text()})
            json_response = json.loads(response.text)

            # HTTP 200
            if response.ok:
                # Open settings window
                self.open_settings_window(json_response['token'])
            # Bad response
            else:
                # Show error
                self.popup.setText("Username or Password is not correct")
                self.popup.exec_()
        except Exception as e:
            # Unable to access server
            self.popup.setText(f"Unable to access server: {e}")
            self.popup.exec_()

    def open_settings_window(self, token):
        self.settings_window = SettingsWindow(token)
        self.settings_window.displayInfo()
        self.close()

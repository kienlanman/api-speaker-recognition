# -*- coding: utf-8 -*-
from src.api import app

if __name__ == '__main__':
    # Running app in debug mode
    app.run(debug=True, host='192.168.1.8')
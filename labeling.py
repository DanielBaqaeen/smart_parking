from git import Repo
import os


def labeling():
    # only first tim
    # os.system('pip install --upgrade pyqt5 lxml')

    # Image labeling
    label_img_path = "labelImg"
    # check if the repo exists
    # Repo.clone_from("https://github.com/tzutalin/labelImg", label_img_path)

    os.system('cd ' + label_img_path + ' && pyrcc5 -o libs/resources.py resources.qrc')
    os.system('cd ' + label_img_path + ' && python labelImg.py')


labeling()

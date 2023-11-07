import os
import pathlib


def main():
    path = str(pathlib.Path().resolve()) + '/traffic_lights/train/images/'
    directory = os.fsencode(path)
    count = 1

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        filename, extension = os.path.splitext(filename)
        if extension.endswith(".jpg") or extension.endswith(".jpeg") or extension.endswith(".png"):
            # print(path + 'img_' + str(count) + extension)
            os.rename(path + filename + extension, path + 'img_' + str(count) + extension)
            count += 1
            continue
        else:
            continue


if __name__ == '__main__':
    main()

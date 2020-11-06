import os


def uniquify_filename(path):
    """
    Use this method to make a filename unique. If another file with the same name exists, this method return a
    new filename with a number between brackets.
    Es: if the name "./result.txt" already exists, the path returned will be "./results(1).txt"
    :param path: filename path to uniquify
    :return: uniquified new filename path
    """
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = f"{filename}({counter}){extension}"
        counter += 1

    return path

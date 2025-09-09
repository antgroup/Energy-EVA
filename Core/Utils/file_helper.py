import hashlib
import os

# glob all satisfied extension files in directory
def glob(_directory, _available_ext=None):
    to_return = []
    for root, directories, files in os.walk(_directory):
        for m_file in files:
            m_file_name, m_file_ext = os.path.splitext(m_file)
            if _available_ext is None or \
                    (_available_ext is not None and m_file_ext.lower() in _available_ext):
                to_return.append([os.path.join(root, m_file), m_file_name, m_file_ext])
    return to_return


def get_file_md5(_filepath):
    m = hashlib.md5()
    with open(_filepath, 'rb') as to_read:
        while True:
            data = to_read.read(4096)
            if not data:
                break
            m.update(data)

    return m.hexdigest()

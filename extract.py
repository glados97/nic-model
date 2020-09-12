import sys
import os
import zipfile

def main(argv):
    os.system('curl -H "Authorization: Bearer ' + argv + '" https://www.googleapis.com/drive/v3/files/16jNwTdwtFXoW_gsxH87TntWh6ICcFzIj?alt=media -o data.zip')
    with zipfile.ZipFile('data.zip', 'r') as zip_ref:
        zip_ref.extractall("nic")

if __name__ == "__main__":
    main(sys.argv[1])
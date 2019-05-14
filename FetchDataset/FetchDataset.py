from google_images_download import google_images_download
import os

def main():
    response = google_images_download.googleimagesdownload()
    

    chromedriver = "C:\chromedriver_win32\chromedriver.exe"

    arguments = {"keywords":"lavender","limit":600,"output_directory":"dataset","chromedriver":chromedriver}

    paths = response.download(arguments)

main()
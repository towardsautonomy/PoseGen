from google_images_search import GoogleImagesSearch

# you can provide API key and CX using arguments,
# or you can set environment variables: GCS_DEVELOPER_KEY, GCS_CX
gis = GoogleImagesSearch("AIzaSyD1Cn-pyVlR5btDnxcxA6MRCFE7RkKw_9M", "46f494dcf51c346e3")

# define search params:
# _search_params = {
#     'q': '...',
#     'num': 10,
#     'safe': 'high|medium|off',
#     'fileType': 'jpg|gif|png',
#     'imgType': 'clipart|face|lineart|news|photo',
#     'imgSize': 'huge|icon|large|medium|small|xlarge|xxlarge',
#     'imgDominantColor': 'black|blue|brown|gray|green|orange|pink|purple|red|teal|white|yellow',
#     'imgColorType': 'color|gray|mono|trans',
#     'rights': 'cc_publicdomain|cc_attribute|cc_sharealike|cc_noncommercial|cc_nonderived'
# }

_search_params = {
    "q": "mustang gt orange",
    "num": 10,
    "safe": "medium",
    "fileType": "jpg|gif|png",
    "imgType": "photo",
    "imgSize": "imgSizeUndefined",  #'huge|large|medium|xlarge|xxlarge',
    "imgDominantColor": "imgDominantColorUndefined",  #'black|blue|brown|gray|green|orange|pink|purple|red|teal|white|yellow',
    "imgColorType": "color",
    "rights": "cc_publicdomain|cc_attribute|cc_sharealike|cc_noncommercial|cc_nonderived",
}

# # this will only search for images:
# gis.search(search_params=_search_params)

# this will search and download:
gis.search(search_params=_search_params, path_to_dir="google/")

# # this will search, download and resize:
# gis.search(search_params=_search_params, path_to_dir='/path/', width=500, height=500)

# # search first, then download and resize afterwards:
# gis.search(search_params=_search_params)
# for image in gis.results():
#     image.download('/path/')
#     image.resize(500, 500)

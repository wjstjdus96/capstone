# 1. File name utillGetCommentExample.py -> utillGetComment.py
# 2. pip install google-api-python-client

from googleapiclient.discovery import build

YOUTUBE_API_KEY = "AIzaSyDOGpulD-gvPmhS0GkmwJe9zQ9Jg0__7QI"

#댓글 가져오기
def getComments(urlvalue) :
    DEVELOPER_KEY = YOUTUBE_API_KEY
    YOUTUBE_API_SERVICE_NAME = "youtube"
    YOUTUBE_API_VERSION = "v3"
    reviews = []
    npt = ""
    videoId_init = urlvalue
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)
    cm = youtube.commentThreads().list(
        videoId=videoId_init,
        order="relevance",
        part="snippet",
        maxResults=100,
        pageToken=npt
    ).execute()

    if 'nextPageToken' in cm.keys():
        while 'nextPageToken' in cm.keys():
            cm = youtube.commentThreads().list(
                videoId=urlvalue,
                order="relevance",
                part="snippet",
                maxResults=100,
                pageToken=npt
            ).execute()
            for i in cm['items']:
                reviews.append(i['snippet']['topLevelComment']['snippet']['textOriginal'])

            if 'nextPageToken' in cm.keys():
                npt = cm['nextPageToken']
            else:
                break
    else:
        for i in cm['items']:
            reviews.append(i['snippet']['topLevelComment']['snippet']['textOriginal'])

    return reviews

# 영상 제목, 썸네일 가져오기
def getThumbnail(urlvalue):
    global title
    global image
    DEVELOPER_KEY = YOUTUBE_API_KEY
    YOUTUBE_API_SERVICE_NAME = "youtube"
    YOUTUBE_API_VERSION = "v3"
    # titles
    # images
    videoId_init = urlvalue
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)
    cm = youtube.videos().list(
        id=videoId_init,
        part='snippet'
    ).execute()

    for i in cm['items']:
        title = i['snippet']['title']
        image = i['snippet']['thumbnails']['medium']['url']
        # image = i['snippet']['thumbnails']['default']['url']

    return title, image
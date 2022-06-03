from django.shortcuts import render

from tubeana.utillGetComment import getComments,getThumbnail
from django.views.decorators.csrf import csrf_exempt
from django import template


strbun = template.Library()


def index(request):
    return render(request, 'tubeana/index.html')


@csrf_exempt
@strbun.filter("board")
def board(request):
    global reviews
    global videoId

    videoId = request.GET.get("url").split("=")[-1]
    print("영상의 아이디 : " + videoId)

    reviews = getComments(videoId)
    title, thumbnail = getThumbnail(videoId)

    from tubeana.predict_views import keyword, percent, top_text, low_text
    exec(open('tubeana/predict_views.py', encoding="UTF-8").read())

    print(keyword)
    context = {
        'videoId': videoId,
        'title': title,
        'thumbnail': thumbnail,
        # 'title' : '제목',
        'percent': percent,
        'top5_text': top_text,
        'low5_text': low_text,
        'keyword': keyword
    }

    return render(request, 'tubeana/board.html', context)


# predict_views.py 로 reviews 를 보낼 때, 사용하는 함수
def send_reviews():
    return reviews


# predict_views.py 로 id 를 보낼 때, 사용하는 함수
def send_id():
    return id

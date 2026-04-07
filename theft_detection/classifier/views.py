from django.shortcuts import render
from .models import Video
from .ai_model import predict_video

def upload_page(request):
    result = None

    if request.method == 'POST':
        video_file = request.FILES['video']
        video = Video.objects.create(file=video_file)

        # تشغيل الموديل
        result = predict_video(video.file.path)

        video.prediction = result
        video.save()

    return render(request, 'upload.html', {'result': result})
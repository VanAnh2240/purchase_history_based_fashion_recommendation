import kagglehub

path = kagglehub.competition_download(
    "h-and-m-personalized-fashion-recommendations"
)

print(path)
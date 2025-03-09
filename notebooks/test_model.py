from ultralytics import YOLO

model = YOLO('../artifacts/models/best.pt')
results = model('../samples/sample1.png')
results.print()
results.show()
results.save()
print(results)

# plot results
results[0].show()
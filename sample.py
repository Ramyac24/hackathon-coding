from ultralytics import YOLO 
model = YOLO("yoloe-11s-seg.pt")
names = [
    "watch",          
    "over-ear headphones", 
    "audio headset",       
    "earbuds",            
    "head-mounted audio device",
    "desk",
    "chair" 
]
text_pe = model.get_text_pe(names)
model.set_classes(names, text_pe)

results = model.predict(
    source="sample.mp4", 
    show=True, 
    conf=0.05, 
    device="cpu"
)
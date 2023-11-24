from time import time
import requests

class YOLOv2:
    model_path = "./model-91934.awnn.mud"
    labels = ['Pollos']
    anchors = [0.63, 0.59, 1.28, 0.94, 1.5, 1.5, 2.28, 2.19, 0.91, 1.12]
    
    def __init__(self) -> None:
        from maix import nn
        self.model = nn.load(self.model_path)
        from maix.nn import decoder
        self.decoder = nn.decoder.Yolo2(len(self.labels), self.anchors, net_in_size=(448, 448), net_out_size=(448 // 32, 448 // 32))
    
    def __del__(self):
        del self.model
        del self.decoder
    
    def cal_fps(self, start, end):
        one_second = 1
        one_flash = end - start
        fps = one_second / one_flash
        return fps

    def draw(self, img, box, probs, fps, class_id):
        from maix import image
        img.draw_rectangle(box[0], box[1], box[0] + box[2], box[1] + box[3], color=(255, 255, 255), thickness=2)
        img.draw_string(0, 0, 'FPS: '+str(fps), scale = 2, color = (0, 0, 255), thickness = 2)

        x=len(probs)
        msg2 = "{}:{:}".format(self.labels[class_id], x)
        w, h = image.get_string_size(msg2, scale=2.4, thickness=2)
        img.draw_string((img.width - w) // 2, img.height // 2 - h // 2, msg2 , scale = 5.5, color = (255, 0, 0), thickness = 14)
        
#        import requests
        payload = {'new_N': x}
        response = requests.post('http://contadorpollos.000webhostapp.com/update.php', data=payload)
    
    def run(self, input):
        t = time()
        out = self.model.forward(input, quantize=1, layout="hwc")
        boxes, probs = self.decoder.run(out, nms=0.3, threshold=0.5, img_size=(448,448))
        for i, box in enumerate(boxes):
            class_id = probs[i][0]
            fps = self.cal_fps(t, time())
            self.draw(input, box, probs, fps, class_id)
        
def main():
    from maix import camera, display
    yolov2 = YOLOv2()

    while True:
        img = camera.capture().resize(size=(448,448))        
        yolov2.run(img)        
        display.show(img)
        
main()


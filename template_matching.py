import numpy as np
import cv2


class TemplateMatch:
    def __init__(self, reference, p):
        self.reference = reference
        self.p = p

    def first_exhaust(self, frame):

        pos_x = 0
        pos_y = 0
        cur_max = -9999
        filtered = cv2.filter2D(frame, -1 ,self.reference)
        frame_height, frame_width = np.shape(filtered)
        for i in range(frame_height):
            for j in range(frame_width):
                if filtered[i][j] > cur_max:
                    pos_x = i
                    pos_y = j
                    cur_max = filtered[i][j]
        return pos_x, pos_y

    def exhaust_search(self, frame, pos_x, pos_y):
        cur_max = -99999
        filtered = cv2.filter2D(frame, -1, self.reference)
        frame_height, frame_width = np.shape(filtered)
        n_search=0
        for i in range(pos_x - self.p, pos_x + self.p + 1):
            for j in range(pos_y - self.p, pos_y + self.p+1):
                if 0 <= i < frame_height and 0 <= j < frame_width:
                    if filtered[i][j]>cur_max:
                        pos_x = i
                        pos_y = j
                        cur_max = filtered[i][j]
                    n_search+=1

        return pos_x, pos_y, n_search

    def log2Dsearch(self, frame, pos_x, pos_y):
        p=self.p
        cur_max = -99999
        filtered = cv2.filter2D(frame, -1, self.reference)
        frame_height, frame_width = np.shape(filtered)
        n_search = 0
        while True:
            k = int(np.ceil(np.log2(p)))
            d = 2 ** (k - 1)
            # print("d: ", d)
            if d <= 1:
                break
            n_tot=0
            for i in range(pos_x-d, pos_x+d+1, d):
                for j in range(pos_y-d, pos_y+d+1, d):
                    if 0 <= i < frame_height and 0 <= j < frame_width:
                        if filtered[i][j] > cur_max:
                            pos_x = i
                            pos_y = j
                            cur_max = filtered[i][j]
                        n_search += 1
                    n_tot+=1
            # print("n_tot: ",n_tot)
            p //= 2

        return pos_x, pos_y, n_search

    def hierarchy_search(self, frame, pos_x, pos_y):
        n_search = 0
        new_size = (np.shape(frame)[0] // 2, np.shape(frame)[1]//2)
        frame1 = cv2.GaussianBlur(frame,(5,5),0)
        frame1 = cv2.resize(frame1, new_size)

        new_size = (np.shape(self.reference)[0] // 2, np.shape(self.reference)[1] // 2)
        reference1 = cv2.GaussianBlur(self.reference,(5,5),0)
        reference1 = cv2.resize(reference1, new_size)

        new_size = (np.shape(frame1)[0] // 2, np.shape(frame1)[1] // 2)
        frame2 = cv2.GaussianBlur(frame1,(5,5),0)
        frame2 = cv2.resize(frame2, new_size)

        new_size = (np.shape(reference1)[0] // 2, np.shape(reference1)[1] // 2)
        reference2 = cv2.GaussianBlur(reference1, (5, 5), 0)
        reference2 = cv2.resize(reference2, new_size)

        x=pos_x
        y=pos_y

        model1 = TemplateMatch(reference2, self.p//4)
        opt_x, opt_y, srch = model1.exhaust_search(frame2, x//4, y//4)
        n_search+=srch

        x1 = opt_x-x//4
        y1 = opt_y-y//4
        pos_x = x//2 + 2 * x1
        pos_y = y//2 + 2 * y1

        model2 = TemplateMatch(reference1, 1)
        opt_x, opt_y, srch = model2.exhaust_search(frame1, pos_x, pos_y)
        n_search += srch

        x2 = opt_x - x//2
        y2 = opt_y - y//2

        pos_x = x + 2 * x2
        pos_y = y + 2 * y2

        model3 = TemplateMatch(self.reference, 1)
        opt_x, opt_y, srch = model3.exhaust_search(frame, pos_x, pos_y)
        n_search += srch
        return opt_x, opt_y, n_search




res= open("./template_matching/result.txt","w")
all_p=[7]
methods=[2]

p=10
method =3

for p in all_p:
    for method in methods:
        print("p:",p," method: ",method)
        reference = cv2.imread('./template_matching/reference.jpg')
        reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

        cap = cv2.VideoCapture('./template_matching/movie.mov')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('./template_matching/output.mov', fourcc, fps, (width, height))

        model = TemplateMatch(reference = reference, p=p)
        outframes = []
        id=0
        pos_x = 0
        pos_y = 0
        total_search=0
        while cap.isOpened():
            # print("Frame: ",id)
            ret, org_frame = cap.read()
            if ret == False:
                cap.release()
                break
            frame = cv2.cvtColor(org_frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.bitwise_not(frame)
            frame = (frame * 0.0002)

            if id==0:
                pos_x,pos_y = model.first_exhaust(frame)
            else:
                if method==1:
                    pos_x, pos_y, n_search = model.exhaust_search(frame, pos_x=pos_x, pos_y=pos_y)
                    total_search += n_search
                elif method==2:
                    pos_x, pos_y, n_search = model.log2Dsearch(frame, pos_x=pos_x, pos_y=pos_y)
                    total_search += n_search
                elif method==3:
                    pos_x, pos_y, n_search = model.hierarchy_search(frame, pos_x=pos_x, pos_y=pos_y)
                    total_search += n_search
                else:
                    print("select valid method!!")
                    break

            ref_height, ref_width = np.shape(reference)
            outframe=np.ones((height,width,3),np.uint8)*255

            start_x = pos_x - int(ref_height / 2)
            start_y = pos_y - int(ref_width / 2)
            end_x = pos_x + int(ref_height / 2)
            end_y = pos_y + int(ref_width / 2)

            cv2.rectangle(org_frame, (start_y, start_x), (end_y, end_x), (255, 0, 0), 2)
            # cv2.imwrite('./template_matching/frame'+str(id)+'.jpg', outframe)
            out.write(org_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            id+=1

        cap.release()
        out.release()

        result= "method: "+str(method)+" avererage num of search: "+str(total_search/id)+ " for p=" +str(p)+"\n"
        res.write(result)
res.close()
# fork_spoon
1. Required:
* pytorch  
* opencv  
* numpy  
* CNN_fs.pt  
Download the CNN_fs.pt in the master branch, put it into the folder the same as test_fs.py.  
  
2. Change the path of 
* 99th line, load the CNN_fs.pt file:  
model_state_dict = torch.load("your file path")
* 115th line, open a video file:  
cam = cv2.VideoCapture("your file path")
* 120th line, save the result:  
outVideo = cv2.VideoWriter("your file path", four_cc, fps, size)

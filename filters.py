from argparse import ArgumentParser
from cv2 import cv2 as cv
import random
import os
import numpy as np

def main():
  if os.path.exists('outputs')==False: # generating file system for outputting results,only runs on first execution of code
    try:
      os.mkdir('outputs')
    except OSError:
      print("Directory creation failed")
  
  if os.path.exists('outputs/problem1')==False:
    try:
      os.mkdir('outputs/problem1')
    except OSError:
      print("Directory creation failed")

  if os.path.exists('outputs/problem2')==False:
    try:
      os.mkdir('outputs/problem2')
    except OSError:
      print("Directory creation failed")

  if os.path.exists('outputs/problem3')==False:
    try:
      os.mkdir('outputs/problem3')
    except OSError:
      print("Directory creation failed")

  if os.path.exists('outputs/problem4')==False:
    try:
      os.mkdir('outputs/problem4')
    except OSError:
      print("Directory creation failed")

  args=parse_arguments()
  img=cv.imread(args.input_image) # Command Line Arguments are parsed

  if args.problem==1: # Problem 1
    img=problem1(img,args.darkening_coefficient,args.blending_coefficient,args.evaporation_coefficient,args.pos,args.leak_thickness,args.mode1)
  elif args.problem==2: # Problem 2
    img=problem2(img,args.blending_coefficient,args.kernel_size,args.g_factor,args.green_value,args.rb_noise_ratio,args.mode2)
  elif args.problem==3: # Problem 3
    img=problem3(img,args.blurring_amount,args.darkening_coefficient,args.is_Sharpen,args.is_LUT,args.is_Hist)
  elif args.problem==4: # Problem 4
    img=problem4(img,args.swirl_strength,args.swirl_radius,args.swirl_rotation,args.blurring_amount)

  cv.imwrite(args.output_file,img)

  return img

def parse_arguments():
  parser=ArgumentParser() # arguments are parsed using the parser
  parser.add_argument('-pn','--problem',
                      type=int,
                      default=1,
                      const=1,
                      nargs='?',
                      choices=[1,2,3,4],
                      help='Problem number')
  parser.add_argument('-i','--input_image',
                      type=str,
                      default='face1.jpg',
                      help='Input Image')
  parser.add_argument('-o','--output_file',
                      type=str,
                      default='output.jpg',
                      help='Output Image')
  parser.add_argument('-ev','--evaporation_coefficient',
                      type=float,
                      default=0.9,
                      help='The evaporation factor(light leak)')
  parser.add_argument('-p','--pos',
                      type=float,
                      default=0.4,
                      help='The vertical position of light leak')
  parser.add_argument('-lt','--leak_thickness',
                      type=float,
                      default=0.17,
                      help='The thickness of the leak')
  parser.add_argument('-dc','--darkening_coefficient',
                      type=float,
                      default=1.2,
                      help='The darkening coefficient')
  parser.add_argument('-bc','--blending_coefficient',
                      type=float,
                      default=0.5,
                      help='The blending coefficient')
  parser.add_argument('-ks','--kernel_size',
                      type=int,
                      default=15,
                      help='The kernel size')
  parser.add_argument('-g','--g_factor',
                      type=float,
                      default=0.4,
                      help='The gradient')
  parser.add_argument('-gV','--green_value',
                      type=int,
                      default=0,
                      help='The green value for coloured pencil filter')
  parser.add_argument('-rbR','--rb_noise_ratio',
                      type=float,
                      default=0.5,
                      help='Blending ratio between noise textures for red and blue')
  parser.add_argument('-br','--blurring_amount',
                      type=float,
                      default=0.5,
                      help='The blurring amount')
  parser.add_argument('-iS','--is_Sharpen',
                      type=str,
                      default="F",
                      help='Sharpen or not')
  parser.add_argument('-iL','--is_LUT',
                      type=str,
                      default="T",
                      help='Apply LUT or not')
  parser.add_argument('-iH','--is_Hist',
                      type=str,
                      default="F",
                      help='Apply HE or not')
  parser.add_argument('-ss','--swirl_strength',
                      type=int,
                      default=2,
                      help='The swirl strength')
  parser.add_argument('-srad','--swirl_radius',
                      type=int,
                      default=100,
                      help='The swirl radius')
  parser.add_argument('-srot','--swirl_rotation',
                      type=int,
                      default=0,
                      help='The swirl rotations')
  parser.add_argument('-m1','--mode1',
                      type=str,
                      default="simple",
                      const="simple",
                      nargs="?",
                      choices=["simple","rainbow"],
                      help='Mode for Problem 1')
  parser.add_argument('-m2','--mode2',
                      type=str,
                      default="mono",
                      const="mono",
                      nargs="?",
                      choices=["mono","colour"],
                      help='Mode for Problem 2')

  return parser.parse_args()

def problem1(img,darkening_coefficient,blending_coefficient,evaporation_coefficient,pos,leak_thickness,mode):
  if boundcheck(darkening_coefficient,1.0,0.0)==False: # This segment is to check for out of bounds variables and adjust them
    darkening_coefficient=0.5
  if boundcheck(blending_coefficient,1.0,0.0)==False:
    blending_coefficient=0.5
  if boundcheck(evaporation_coefficient,1.0,0.0)==False:
    evaporation_coefficient=0.9
  if boundcheck(pos,0.8,0.0)==False:
    pos=0.4
  if boundcheck(leak_thickness,0.49,0.0)==False:
    leak_thickness=0.17
  if leak_thickness+pos>1.0:
    pos=0.4

  rows,cols=img.shape[0],img.shape[1]
  leak=[] # Array holds 'light' and is written is point transformed to each row of the mask
  mask = np.zeros((rows,cols, 3), np.uint8)
  mask[:] = (50,50,50) # initial mono coloured mask
  pos=int(pos*rows)
  leak_len=int(leak_thickness*rows)
  leak=np.zeros((1,(leak_len),3),np.uint8)

  img=blend(img,1-darkening_coefficient,mask,darkening_coefficient,rows,cols) # input image is darkened here

  if mode == "simple":
    leak[:] = (255,255,255) # Fully white leak for simple mode
  else:
    leak=rainbow(leak,leak_len,0) # rainbow leak is created
  
  r1=0
  leak_star=0
  dire=[]
  if random.randint(1,2)==2: # randomly decides direction of leak
    dire=[1,10,0]
  else:
    dire=[0,10,1]

  for i in range(1,cols-1):
    for j in range(leak_star,leak_len):
      mask[i,pos+j,:]=leak[0][j] # leak is pasted onto the mask from pos to pos+leak_len

    r1=random.choices([1,2,3],dire,k=1)[0]
    
    if r1==1: # randomly decides whether or not to shift direction
      dire[0]+=0.1
      dire[1]+=0.1
      pos-=1
      mask[i,pos,:]=mask[i,pos+1,:]
    elif r1==3:
      dire[2]+=0.1
      dire[1]+=0.1
      pos+=1
    
    if i+leak_len<rows:
      if mode=="simple":
        if random.randint(0,1000)>evaporation_coefficient*1000: # randomly decides whether the leak size shrinks
          leak_len-=1

  if mode=="simple":
    mask=img_divide(mask,rows,cols,2.0) # darkens the mask
    mask=blur(mask,rows,cols,1,21) # apply blur for realism
    cv.imwrite("outputs/problem1/simple_mask.jpg",mask)
    blended=blend(img,blending_coefficient,mask,1-blending_coefficient,rows,cols) # Blending,similar to addweighted
    cv.imwrite("outputs/problem1/simple_leak.jpg",blended)
    return blended
  else:
    mask=img_divide(mask,rows,cols,1.15) # darkens the mask
    mask=blur(mask,rows,cols,1,21) # apply blur for realism
    cv.imwrite("outputs/problem1/rainbow_mask.jpg",mask)
    blended=blend(img,blending_coefficient,mask,1-blending_coefficient,rows,cols) # Blending,similar to addweighted
    cv.imwrite("outputs/problem1/rainbow_leak.jpg",blended)
    return blended

def problem2(img,blending_coefficient,kernel_size,g_factor,green_value,rb_noise_ratio,mode):
  if boundcheck(blending_coefficient,1.0,0.0)==False: # Validation
    blending_coefficient=0.3

  rows,cols=img.shape[0],img.shape[1]
  gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY) # converting image to grayscale
  k=(0.4)*avg_col(gray,0,0,rows,cols) # assigning value to k

  if mode=="mono": # for 'mono' mode
    if boundcheck(g_factor,1.0,0.0)==False: # Validation
      g_factor=0.6
    if boundcheck(kernel_size,cols+rows/20,5)==False:
      kernel_size=15

    tex=np.zeros((rows,cols,1), np.uint8) # initializing noise texture
    kernel=np.zeros((kernel_size,kernel_size)) # initializing kernel for motion blur

    for i in range(kernel_size): # kernel is filled roughly diagonally 
      for j in range(kernel_size):
        if i==j:
          kernel[i,j]=1
          if j!= kernel_size-1:
            kernel[i,j+1]=1
            if j!=kernel_size-2:
              kernel[i][j+2]=1 
          if i!=kernel_size-1:
            kernel[i+1][j]=1
            if i!=kernel_size-2:
              kernel[i+2][j]=1

    kernel/=kernel_size
    # For kernel size 5 : kernel=
    #  [0.2 0.2 0.2 0.  0. ]
    #  [0.2 0.2 0.2 0.2 0. ]
    #  [0.2 0.2 0.2 0.2 0.2]
    #  [0.  0.2 0.2 0.2 0.2]
    #  [0.  0.  0.2 0.2 0.2]

    for i in range(0,rows): # random noise in the range k-75 to k+25 is applied to create the initial texture
      for j in range(0,cols):
        if k-75>0:
          tex[i,j]=random.randint(int(k-75),int(k+25))
        else:
          tex[i,j]=random.randint(0,int(k+25))

    tex=img_divide(tex,rows,cols,1.5) # texture is darkened
    tex=cv.filter2D(tex,-1,kernel) # motion blur/convolutions are applied
    gradient=0

    for i in range(0,rows): # brightness gradient is applied
      for j in range(0,cols):
        gradient=((i*j)+10)/(cols*rows/1.5)
        tex[i,j]*=gradient**(g_factor)

    cv.imwrite("outputs/problem2/mono_pencil_noise.jpg",tex)
    pencil=blend(gray,blending_coefficient,tex,1-blending_coefficient,rows,cols)

    for i in range(0,rows): # brightness gradient is applied again
      for j in range(0,cols):
        gradient=((i*j)+30)/(cols*rows/1.5)
        pencil[i,j]*=gradient**g_factor
    cv.imwrite("outputs/problem2/monochrome_pencil.jpg",pencil)
    return pencil
  else: # id mode = 'colour'
    if boundcheck(rb_noise_ratio,1.0,0.0)==False: # validation
      rb_noise_ratio=0.55
    if green_value>255:
      green_value=255
    elif green_value<0:
      green_value=0

    colr= np.zeros((rows,cols, 3), np.uint8) # initializing colour image
    for i in range(0,rows):
      for j in range(0,cols):
        colr[i,j]=(gray[i,j],gray[i,j],gray[i,j]) # conversion from grayscale to bgr

    redh = np.zeros((rows,cols, 3), np.uint8) # 'red' noise texture is initialized
    redh[:] = (0,green_value,205)
    for i in range(0,rows): # random noise is generated for the red texture
      for j in range(0,cols):
        redh[i,j,2]=random.randint(redh[i,j,2]-100,redh[i,j,2]+50)

    cv.imwrite("outputs/problem2/red_noise.jpg",redh)
    blueh = np.zeros((rows,cols, 3), np.uint8) # 'blue' noise texture is initialized
    blueh[:] = (125,green_value,0)
    for i in range(0,rows): # random noise is generated for the blue texture
      for j in range(0,cols):
        blueh[i,j,2]=random.randint(blueh[i,j,2]-75,blueh[i,j,2]+130)

    blueh=blur(blueh,rows,cols,1.0,7) # blur is applied to the blue noise texture
    cv.imwrite("outputs/problem2/blue_noise.jpg",blueh)

    hue = blend(redh,rb_noise_ratio,blueh,1-rb_noise_ratio,rows,cols) # red and blue textures are blended 
    cv.imwrite("outputs/problem2/colo_pencil_noise.jpg",hue)

    colr=blend(colr,blending_coefficient,hue,1-blending_coefficient,rows,cols) # resulting noise texture is blended with the grayscale image
    cv.imwrite("outputs/problem2/coloured_pencil.jpg",colr)

    return colr

def problem3(img,blurring_amount,darkening_coefficient,is_Sharpen,is_LUT,is_Hist):
  if boundcheck(darkening_coefficient,1.0,0.0)==False: # Validation
    darkening_coefficient=0.0
  if boundcheck(blurring_amount,1.5,0.0)==False:
    blurring_amount=0.4

  rows,cols=img.shape[0],img.shape[1]
  smooth=blur(img,rows,cols,blurring_amount,3) # image smoothing

  cv.imwrite("outputs/problem3/blur.jpg",smooth)
  beautified=smooth.astype(np.uint8)
  beautified=img_divide(beautified,rows,cols,1+darkening_coefficient) # image darkening
  if is_LUT=="T": # LUT option
    rsum,bsum=0,0
    for i in range(0,rows): # calculating the red and blue averages of the image
      for j in range(0,cols):
        rsum+=smooth[i,j,2]
        bsum+=smooth[i,j,0]

    r_channel,b_channel,g_channel=[],[],[] # initializing LUT channels
    if rsum>bsum/2: # Warm image
      r_channel=lut_channel(np.zeros(256, np.dtype('uint8')),255)
      b_channel=lut_channel(np.zeros(256, np.dtype('uint8')),195)
      g_channel=lut_channel(np.zeros(256, np.dtype('uint8')),195)
    else: # Cool image
      r_channel=lut_channel(np.zeros(256, np.dtype('uint8')),255)
      b_channel=lut_channel(np.zeros(256, np.dtype('uint8')),195)
      g_channel=lut_channel(np.zeros(256, np.dtype('uint8')),195)

    lut = np.dstack((b_channel,g_channel,r_channel)) # LUT is created
    cv.imwrite("outputs/problem3/lut.jpg",lut)
    beautified=cv.LUT((beautified).astype(np.uint8),lut) # LUT is applied

  if is_Hist=="T":
    ycc=cv.cvtColor(beautified,cv.COLOR_BGR2YCrCb) # conversion to YCbCR
    cv.imwrite("outputs/problem3/YCbCr.jpg",ycc)
    hist=ycc[:,:,0] # Y channel
    hist=cv.equalizeHist(hist) # Histogram is equalized
    cv.imwrite("outputs/problem3/Histogram.jpg",hist)
    for i in range(0,rows):
      for j in range(0,cols):
        ycc[i,j,0]=hist[i,j]
    beautified=cv.cvtColor(ycc,cv.COLOR_YCrCb2BGR) # Back to BGR

  if is_Sharpen=="T": # image sharpening
    kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    beautified=cv.filter2D(beautified,-1,kernel)

  cv.imwrite("outputs/problem3/beautified.jpg",beautified)
  return beautified

def problem4(img,strength,radius,rotation,blurring_amount):
  if boundcheck(blurring_amount,1.5,0.0)==False: # Validation
    blurring_amount=0.9

  rows,cols=img.shape[0],img.shape[1]
  eyeX=0.5 * float(rows-1.0) # X coord of center
  eyeY=0.5 * float(cols-1.0) # Y coord of center

  swirled=np.zeros((rows,cols,3))
  for x in range(0,rows): # swirling an image without low pass filter using bilinear unterpolation
    for y in range(0,cols):
      rho=np.sqrt((x-eyeX)**2 + (y-eyeY)**2) # rho is distance from centre
      dispersion=radius**1.75/rho # dispersion increases with lower rho, there will be more swirl for image closer to the center
      theta = rotation + strength * np.exp(-rho / dispersion) + np.arctan2(y - eyeY, x - eyeX) # the angle
      sX=int(eyeX+rho*np.cos(theta)) # sX = eyeX if x is outside radius, otherwise itt is shifted
      sY=int(eyeY+rho*np.sin(theta)) # sY = eyeY if y is outside radius, otherwise itt is shifted
      if sX>=0 and sX<rows and sY>=0 and sY<cols:
        swirled[x,y]=img[sX,sY] # Apply reverse mapping

  cv.imwrite('outputs/problem4/swirl_without_lpf.jpg',swirled)
  inverse=np.zeros((rows,cols,3))

  for x in range(0,rows): # reverting image back to original
    for y in range(0,cols):
      rho=np.sqrt((x-eyeX)**2 + (y-eyeY)**2)
      dispersion=radius**1.75/rho
      theta = rotation + strength * np.exp(-rho/dispersion) + np.arctan2(y-eyeY,x-eyeX)
      sX=int(eyeX+rho*np.sin(theta))
      sY=int(eyeY+rho*np.cos(theta))
      if sX>=0 and sX<rows and sY>=0 and sY<cols:
        inverse[y,x]=swirled[sX,sY]

  cv.imwrite('outputs/problem4/reverted_from_swirl_without_lpf.jpg',inverse)
  sub=cv.subtract(img.astype(np.uint8),inverse.astype(np.uint8)) # image subtraction
  cv.imwrite('outputs/problem4/difference_without_lpf.jpg',sub)
  img=blur(img,rows,cols,blurring_amount,3) # low pass filter/mean filter
  cv.imwrite('outputs/problem4/low_pass_filtered.jpg',img)

  swirled=np.zeros((rows,cols,3))
  for x in range(0,rows): # swirling image with low pass filter
    for y in range(0,cols):
      rho=np.sqrt((x-eyeX)**2 + (y-eyeY)**2)
      dispersion=radius**1.75/rho
      theta = rotation + strength * np.exp(-rho / dispersion) + np.arctan2(y- eyeY,x-eyeX)
      sX=int(eyeX+rho*np.cos(theta))
      sY=int(eyeY+rho*np.sin(theta))
      if sX>=0 and sX<rows and sY>=0 and sY<cols:
        swirled[x,y]=img[sX,sY]

  cv.imwrite('outputs/problem4/swirl_with_lpf.jpg',swirled)
  inverse=np.zeros((rows,cols,3))
  for x in range(0,rows): # reverting image
    for y in range(0,cols):
      rho=np.sqrt((x-eyeX)**2 + (y-eyeY)**2)
      dispersion=radius**1.75/rho
      theta = rotation + strength * np.exp(-rho/dispersion) + np.arctan2(y-eyeY,x-eyeX)
      sX=int(eyeX+rho*np.sin(theta))
      sY=int(eyeY+rho*np.cos(theta))
      if sX>=0 and sX<rows and sY>=0 and sY<cols:
        inverse[y,x]=swirled[sX,sY]

  cv.imwrite('outputs/problem4/reverted_from_swirl_with_lpf.jpg',inverse)
  sub=cv.subtract(img,inverse)
  cv.imwrite('outputs/problem4/difference_with_lpf.jpg',sub)
  return swirled

def blur(img,rows,cols,blurring_amount,n): # Mean Filtering
  imgNew = np.zeros((rows, cols, 3))
  for x in range(1, rows-1):
    for y in range(1, cols-1):
        for z in range(3):
          if x>=n and y>=n: # To ensure no section of image gets cut off
            neighbourhood = img[x-n:x+n, y-n:y+n, z]
            imgNew[x,y,z] = (1-blurring_amount)*img[x,y,z] + (blurring_amount)*np.mean(neighbourhood)
          elif x<n and y<n:
            neighbourhood= img[n-x:x+n, n-y:y+n, z]
            imgNew[x,y,z] = (1-blurring_amount)*img[x,y,z] + (blurring_amount)*np.mean(neighbourhood)
          elif x<n:
            neighbourhood= img[n-x:x+n, y-n:y+n, z]
            imgNew[x,y,z] = (1-blurring_amount)*img[x,y,z] + (blurring_amount)*np.mean(neighbourhood)
          elif y<n:
            neighbourhood= img[x-n:x+n, n-y:y+n, z]
            imgNew[x,y,z] = (1-blurring_amount)*img[x,y,z] + (blurring_amount)*np.mean(neighbourhood)
  return imgNew

def img_divide(img,rows,cols,n): # darkens image by dividing it by n
  for i in range(0,rows):
    for j in range(0,cols):
      img[i,j]=img[i,j]/n
  return img

def blend(img1,b1,img2,b2,rows,cols): # blends 2 images, similar to addweighted
  for i in range(0,rows):
    for j in range(0,cols):
      img1[i,j]=(b1*img1[i,j])+(b2*img2[i,j])
  return img1

def rainbow(leak,n,j): # creates a rainbow and stores in a 1D array
  for i in range(0,n):
    if i==0:
      leak[0][i]=(255,0,255) # purple/violet
      j+=1
      continue
    
    leak[0][i]=leak[0][i-1]

    if j<int(0.25*n):
      leak[0][i][2]=leak[0][i][2] - int(1020/(n+0.01))
      j+=1
      continue
    if j==int(0.25*n):
      leak[0][i]=(255,0,0) # Blue

    if j<int(0.5*n):
      leak[0][i][0]=leak[0][i][0] - int(1020/(n+0.01))
      leak[0][i][1]=leak[0][i][1] + int(1020/(n+0.01))
      j+=1
      continue

    if j==int(0.5*n):
      leak[0][i]=(0,255,0) # Green

    if j<int(0.75*n):
      leak[0][i][2]=leak[0][i][2] + int(1020/(n+0.01))
      j+=1
      continue

    if j==int(0.75*n):
      leak[0][i]=(0,255,255) # Yellow
      j+=1

    if leak[0][i][1] != 0:
      leak[0][i][1]=leak[0][i][1] - int(1020/(n+0.01))
    # Red at the end

  return leak

def avg_col(img,avg,n,rows,cols): # returns average colour of an image
  for x in range(0,rows):
    for y in range(0,cols):
      n+=1
      avg+=img[x,y]

  avg/=n
  return avg

def lut_channel(channel,max): # generates a channel for a Look Up table
  for i in range(0,256):
    channel[i]=i*max/256
  return channel

def boundcheck(n,u,l): # used in validating Command Line inputs
  if n>u:
    return False
  elif n<l:
    return False
  return True

if __name__=='__main__':
  main()
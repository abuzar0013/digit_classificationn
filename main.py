import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    if len(physical_devices) > 1:
        tf.config.experimental.set_memory_growth(physical_devices[1], True)
print(tf.__version__)
x = tf.constant([[1,2,3],[4,5,6]])  #marices banane k liye
y = tf.ones((3,3)) # only 1 ki matrices bnane k liye m*n ki
z = tf.zeros((10,10))  # sirf zero ki matrices bnane k liye
k = tf.eye(10) # identity marics bnane k liye
m = tf.range(start = 1, limit=10, dtype = tf.float32) # kisi range me values likhne k liye
b = tf.cast(x, dtype=tf.float32) # humne upar x create kia hai vo innt me create hhua ab usko floa me convert krne k liye
a = tf.random.normal((4,4), mean=0.5, stddev=0.1) #ek matrix create kro jiski values ka mean 0.5 ho aur vo 4*4 ki ho
c = tf.constant([1,2,3])
d = tf.constant([4,5,6])
e =  c+d  # ading two no.
e1 = c-d
e3 =  c*d
e2=  c/d
e4  = tf.tensordot(c,d, axes = 1) # multiply kr k sbko add kr dega
e5 = tf.reduce_sum(c*d, axis=0) # same kaam
e6 = c**5 #sare elements ki power 5 kr dega
e7 =  tf.matmul(z,k) # matrix multiplication
e8 = z @ k #same kaam k liye pr indono k liyye headding me kuch add krna pada hai
g = tf.constant([0,1,2,3,4,5,6,7,8])
print(g[:]) #puri marix prin krane k liye
print(g[1:]) #ek se le kr bakii sara
print(g[1:3]) # ek se teen tak
print(g[::2]) # do do chod kr
print(g[::-1])  #reverse printing
indices = tf.constant([0,4])
g_ind = tf.gather(g, indices)
print(g_ind)
t = tf.constant([[1,2,3],[4,5,6],[4,5,6]])
print(t[2,:]) #second row all column
print(t[0:2,])  # suru ki do row aur column sara


l = tf.constant([0,1,2,3,4,5,6,7,8])
l = tf.reshape(l, (3,3)) #is array jisi matrix ko 3*3 me convert krne k liye
print(l)

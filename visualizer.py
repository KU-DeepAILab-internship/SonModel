alpha = 'A'
for i in range(0,26):
    print("{}:'{}'".format(i,chr(ord(alpha)+i)),end=', ')

classes = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J', 10:'K', 11:'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q', 17:'R', 18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z'}

fig = plt.figure(figsize=(13,13))
for i in range(0, len(history_list)):
    a = fig.add_subplot(10,15,i+1)
    a.set_title('ov : {} \nans : {}'.format(classes[history_list[i].overall.tolist()], classes[history_list[i].ans.tolist()]))
    a.axis('off')
    a.imshow(history_list[i].data.view(32,32))
plt.subplots_adjust(bottom=0.2, top=1.2, hspace=0)
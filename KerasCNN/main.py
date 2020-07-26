from keras.layers import *
from keras.models import *
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
#from math import ceil


from loadFile import data_process
from simple_model import build_simple_model
from model import build_model


import matplotlib.pyplot as plt                        
#matplotlib inline  


if __name__ == '__main__':	
    X_train,X_val,y_train,y_val,X_test,y_test=data_process()
    #simple_model = build_simple_model(10,32)
    #simple_model.summary()
    model = build_model(out_dims=10, img_size=32)
    model.summary()
    
    #对训练集进行数据增强
    datagen = ImageDataGenerator(width_shift_range = 0.2, 
                             height_shift_range = 0.2, 
                             horizontal_flip = True,
                             zoom_range = 0.3) 

    batch_size = 64
    train_generator = datagen.flow(X_train,y_train,batch_size=batch_size, shuffle=False)
    
    
    
    checkpointer = ModelCheckpoint(filepath='cifa_10_model.hdf5', 
                              verbose=1, save_best_only=True) #保存最好模型权重
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    epochs = 20

    #history = model.fit(X_train, y_train,
    #      validation_data=(X_val, y_val),
    #      epochs=epochs,callbacks=[checkpointer],verbose=1)
    history = model.fit_generator(train_generator,
          validation_data = (X_val,y_val),
          epochs=epochs,
          callbacks=[checkpointer],steps_per_epoch=X_train.shape[0],
          verbose=1)
    
    
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
# summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
    
    model.load_weights('cifa_10_model.hdf5')
    y_pred = model.predict(X_test)
    count = 0
    for i in range(len(y_pred)):
        if(np.argmax(y_pred[i]) == np.argmax(y_test[i])): #argmax函数找到最大值的索引，即为其类别
            count += 1
    score = count/len(y_pred)
    print('正确率为:%.2f%s' % (score*100,'%'))
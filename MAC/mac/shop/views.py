from django.shortcuts import render
from .models import Product, Supplies, Orders, OrderUpdate
from math import ceil
import json
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse

def index(request):
    return render(request, 'shop/index.html')

def about(request):
    return render(request, 'shop/about.html')

def supplies(request):
    thank = False
    if request.method=="POST":
        name = request.POST.get('name', '')
        email = request.POST.get('email', '')
        phone = request.POST.get('phone', '')
        desc = request.POST.get('desc', '')
        image = request.POST.get('image', '')
        contact = Supplies(name=name, email=email, phone=phone, desc=desc,image=image)
        contact.save()
        thank = True
    return render(request, 'shop/contact.html')

    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions


    # Mount Google Drive
    from google.colab import drive
    drive.mount('/content/drive')

    # Seting the path to the dataset directory
    dataset_path = '/content/drive/MyDrive/dataset_directory'

    # Defining image size and batch size
    img_size = (224, 224)
    batch_size = 32

    # Data augmentation for training set
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    # Data augmentation for test set (only rescaling)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Loading the training set of a particular batch size
    train_set = train_datagen.flow_from_directory(
        dataset_path + '/train',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    # Loading the test set from the directory 
    test_set = test_datagen.flow_from_directory(
        dataset_path + '/test',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    # Building the CNN model for image classification
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(img_size[0], img_size[1], 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compiling the models developed
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Training the model and validation
    model.fit(train_set, epochs=10, validation_data=test_set)

    # A function to check if the image represents signs of potential violenc in the area.
    def IS_WAR_ZONE(image_path):
    # Loading and preprocessing the input image
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = preprocess_input(img_array)

    # Making the prediction on the inputed data
    prediction = model.predict(img_array)

    ''' defining the result as we used the sigmoid function for the classification. and lowering the threshold  
        considering the gravity of the situation .
    '''
    result = True if prediction >= 0.35 else False
    return result
# The above code is not working but we can use this after making the dataset
def tracker(request):
    if request.method=="POST":
        orderId = request.POST.get('orderId', '')
        email = request.POST.get('email', '')
        try:
            order = Orders.objects.filter(order_id=orderId, email=email)
            if len(order)>0:
                update = OrderUpdate.objects.filter(order_id=orderId)
                updates = []
                for item in update:
                    updates.append({'text': item.update_desc, 'time': item.timestamp})
                    response = json.dumps({"status":"success", "updates":updates, "itemsJson":order[0].items_json}, default=str)
                return HttpResponse(response)
            else:
                return HttpResponse('{"status":"noItem"}')
        except Exception as e:
            return HttpResponse('{"status":"error"}')

    return render(request, 'shop/tracker.html')
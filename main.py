import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import shutil

TRANSFER_LEARNING_MODE =False


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    # load model
    tf.saved_model.loader.load(sess,["vgg16"],vgg_path)
    
    graph = tf.get_default_graph()
    vgg_input_tensor = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer_3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer_4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer_7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return vgg_input_tensor, keep_prob, vgg_layer_3_out, vgg_layer_4_out, vgg_layer_7_out
#tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    
    # avoid bakc propogation through original model layers
    if TRANSFER_LEARNING_MODE :
        vgg_layer3_out = tf.stop_gradient(vgg_layer3_out)
        vgg_layer4_out = tf.stop_gradient(vgg_layer4_out)
        vgg_layer7_out = tf.stop_gradient(vgg_layer7_out)
    
    with tf.variable_scope("decoder"):
        # Up sampling
        # 1 x 1 convolution to reduce features to num_classes
        
        vgg_layer3_out = tf.Print(vgg_layer3_out, [tf.shape(vgg_layer3_out)], message="vgg_layer3_out shape", first_n=1, summarize=4)
        
        vgg_layer4_out = tf.Print(vgg_layer4_out, [tf.shape(vgg_layer4_out)], message="vgg_layer4_out shape", first_n=1, summarize=4)
        
        vgg_layer7_out = tf.Print(vgg_layer7_out, [tf.shape(vgg_layer7_out)], message="vgg_layer7_out shape", first_n=1, summarize=4)
        
        conv_1X1 = tf.layers.conv2d(inputs=vgg_layer7_out, filters=num_classes, kernel_size=1,strides=(1,1), padding='same',
				   kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        
        #print("======== dimensions of conv_1X1 ==========")
        #print(conv_1X1.get_shape())
        upsample1 = tf.layers.conv2d_transpose(inputs=conv_1X1 , filters=num_classes, kernel_size=4,strides=(2,2), padding='same', 
					kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),name="new_transpose1")  
        
        
        pool_4_reshaped = tf.layers.conv2d(inputs=vgg_layer4_out, filters=num_classes, kernel_size=1,strides=(1,1), padding='same',
					kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        
        combine1 = tf.add(upsample1, pool_4_reshaped)

        
        upsample2 = tf.layers.conv2d_transpose(inputs=combine1, filters=num_classes, kernel_size=4,strides=(2,2), padding='same', 
					kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),name="new_transpose2")
        
        pool_3_reshaped = tf.layers.conv2d(inputs=vgg_layer3_out, filters=num_classes, kernel_size=1,strides=(1,1), padding='same',
					kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        
        combine2 = tf.add(upsample2, pool_3_reshaped)
        
        logits = tf.layers.conv2d_transpose(inputs=combine2, filters=num_classes, kernel_size=16,strides=(8, 8),padding='same',
					kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), name="new_logit")
                    
        logits = tf.Print(logits, [tf.shape(logits)], message="Logits shape", first_n=1, summarize=4, name="new_logit")
        
        logits_reshaped = tf.reshape(logits,(-1,num_classes),name="new_logit_reshaped")
        
        logits_reshaped = tf.Print(logits_reshaped, [tf.shape(logits_reshaped)], message="Logits reshaped", first_n=1, summarize=4, name="new_logit_reshaped")
        
        return logits_reshaped
#tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=nn_last_layer,labels=correct_label,name="new_softmax")
    mean_cross_entropy = tf.reduce_mean(cross_entropy)
    
    
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_constant = 0.1  # Choose an appropriate one.
    loss = mean_cross_entropy + reg_constant * sum(reg_losses)
    
    
    correct_label = tf.reshape(correct_label, (-1,num_classes))
    correct_prediction = tf.equal(nn_last_layer, correct_label)
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
    # initialize optimizer 
    # here learning_rate is place holder
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    
    print("TRANSFER_LEARNING_MODE : " , TRANSFER_LEARNING_MODE)
    
    if TRANSFER_LEARNING_MODE :
        trainable_variables = []
        
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "decoder")
        '''
        for variable in tf.trainable_variables():
            # make sure you have named your tensors/ops/layers with names starting with ‘new_’ or some other prefix that you choose
            print(variable.name)
            if "new_" in variable.name or 'beta' in variable.name:
                trainable_variables.append(variable)
        '''
        
        training_op = opt.minimize(loss, var_list=trainable_vars, name="training_op")      
    else :
        training_op = opt.minimize(loss, name="training_op")
    
    
    return nn_last_layer, training_op, accuracy_op, loss
    
#tests.test_optimize(optimize)

def save_model(sess):
   
    save_dir_name = "saved_model_2"
    save_dir = "./"+save_dir_name

    if save_dir_name in os.listdir(os.getcwd()):
        shutil.rmtree(save_dir)

    builder = tf.saved_model.builder.SavedModelBuilder(save_dir)
    builder.add_meta_graph_and_variables(sess, ["vgg16"])
    builder.save()


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label,keep_prob,keep_probb,learning_rate,lr_rate,accuracy_op,get_batches_validation_fn):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function   

    
    for epoch in range(epochs):
        #imgs, labels  = next(get_batches_fn(batch_size))
        for imgs, labels in get_batches_fn(batch_size):
            
            loss, _ = sess.run([cross_entropy_loss,train_op], feed_dict={input_image:imgs, 
                                correct_label:labels,
                                keep_prob:keep_probb,
                                learning_rate :lr_rate
                       })
            print("Loss:", loss)
        
		
        val_imgs, val_lables  = next(get_batches_validation_fn(30))
        accuracy = sess.run([accuracy_op], feed_dict={input_image:val_imgs, 
                                correct_label:val_lables,
                                keep_prob:1.0
                                })
                                
        #print("Accuracy after epoch ", (epoch+1) , ": ", accuracy[0])
        print("Completed epoch ", (epoch+1))
                                
    save_model(sess)
    
#tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    #tests.test_for_kitti_dataset(data_dir)
    lr_rate = float(0.0001)
    epochs=50
    batch_size=20

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        #get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training_small'), image_shape)
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        get_batches_validation_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training_validation'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        # load vgg model
        vgg_input_tensor, keep_prob, vgg_layer_3_out, vgg_layer_4_out, vgg_layer_7_out = load_vgg(sess,vgg_path)
        
        # add new layers in model        
        logits = layers(vgg_layer_3_out, vgg_layer_4_out, vgg_layer_7_out, num_classes)        
                
        #input_image=tf.placeholder(tf.float32,(None,160,576,3),name="input_image")
        correct_label=tf.placeholder(tf.float32,(None,160,576,2),name="correct_label")
        
        learning_rate_ph = tf.placeholder(tf.float32, name='learning_rate')
        
        nn_last_layer, train_op, accuracy_op, mean_cross_entropy = optimize(nn_last_layer=logits, correct_label=correct_label, learning_rate=learning_rate_ph, num_classes=num_classes)
        
        #print(tf.trainable_variables())
          
        
        # initialize variables
        
        if TRANSFER_LEARNING_MODE :
            trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "decoder")
            
            trainable_vars += [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if "beta" in var.name]
            trainable_variable_initializers = [var.initializer for var in trainable_vars]
            sess.run(trainable_variable_initializers)
        else :
            sess.run(tf.global_variables_initializer())
       
        
        # TODO: Train NN using the train_nn function

        keep_probb= float(0.70)
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, mean_cross_entropy, vgg_input_tensor,
             correct_label, keep_prob, keep_probb,learning_rate_ph,lr_rate,accuracy_op,get_batches_validation_fn)
        
        
        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, vgg_input_tensor)
        
        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
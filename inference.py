import os.path
import tensorflow as tf
import helper


def load_mymodel(sess, model_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    final_layer_name = 'decoder/new_logit/BiasAdd:0'
    final_final_layer_name = 'decoder/new_logit_reshaped:0'
    
    # load model
    tf.saved_model.loader.load(sess,["vgg16"],model_path)
    
    node_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
    
    #for name in node_names :
    #    print(name)
    
    graph = tf.get_default_graph()
    
    
    
    input_place_holder = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    logits = graph.get_tensor_by_name(final_layer_name)[0] 
    logits_reshaped = graph.get_tensor_by_name(final_final_layer_name) # [0] not required
    
    logits = tf.Print(logits, [tf.shape(logits)], message="saved logits shape", first_n=1, summarize=4, name="new_logits")
    #logit_reshaped = tf.reshape(logits,(-1,num_classes))
    logits_reshaped = tf.Print(logits_reshaped, [tf.shape(logits_reshaped)], message="saved logits reshape", first_n=1, summarize=4, name="new_logits")
   
    return input_place_holder, keep_prob, logits_reshaped
    
    
    
with tf.Session(graph=tf.Graph()) as sess:
    runs_dir = './runs'
    image_shape = (160, 576)
    data_dir = './data'
    
    print("would load model")
    
    input_place_holder, keep_prob, logits = load_mymodel(sess,'saved_model_2')
    
    print("would use model for inference")
    helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_place_holder)
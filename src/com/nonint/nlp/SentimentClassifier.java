package com.nonint.nlp;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;

import java.nio.IntBuffer;

public class SentimentClassifier {
    public static void main(String[] args) {
        System.out.println(TensorFlow.version());
        try {
            SavedModelBundle bundle = SavedModelBundle.load("C:\\Users\\jbetk\\Documents\\data\\ml\\saved_models\\semantic_seg_amazon_reviews_output_per_star", "serve");
            Session sess = bundle.session();
            int[] zeros = new int[128];
            Tensor[] inputs = new Tensor[3];
            for(int i = 0; i < inputs.length; i++) {
                inputs[i] = Tensor.create(
                        new long[]{ 128 },
                        IntBuffer.wrap(zeros)
                );
            }

            /*
            Iterator<Operation> i = bundle.graph().operations();
            while(i.hasNext()) {
                System.out.println(i.next().name());
            }*/

            float[] res = sess.runner().feed("serving_default_input_ids", inputs[0])
                    .feed("serving_default_attention_mask", inputs[1])
                    .feed("serving_default_token_type_ids", inputs[2])
                    .fetch("dense/bias")
                    .run()
                    .get(0)
                    .copyTo(new float[6]);
            System.out.println(res[0]);
        }catch(Exception e){
            e.printStackTrace();
        }
    }
}

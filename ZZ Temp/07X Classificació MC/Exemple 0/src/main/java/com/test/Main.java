package com.test;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.types.TInt32;
import org.tensorflow.ndarray.buffer.IntDataBuffer;

public class Main {
    public static void main(String[] args) {
        try (Graph graph = new Graph()) {
            Ops tf = Ops.create(graph);
            
            // Create constants using tf.constant()
            var a = tf.constant(3);
            var b = tf.constant(5);
            
            // Add the constants
            var addOp = tf.math.add(a, b);

            // Execute the session to get the result
            try (Session session = new Session(graph)) {
                // Run the operation and get the result
                try (Tensor result = session.runner()
                    .fetch(addOp)
                    .run()
                    .get(0)) {
                    
                    // Get the value from the tensor using DataBuffer with index 0
                    IntDataBuffer buffer = result.asRawTensor().data().asInts();
                    int value = buffer.getInt(0L);  // getInt requires a long index
                    
                    // Print the result
                    System.out.println("Result of 3 + 5 = " + value);
                }
            }
        }
    }
}
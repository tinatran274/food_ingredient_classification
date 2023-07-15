package com.example.ingredientclassification;

import static android.content.ContentValues.TAG;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.ingredientclassification.ml.Model;
import com.example.ingredientclassification.ml.ModelFruit;
import com.example.ingredientclassification.ml.ModelFruits2;
import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.Timestamp;
import com.google.firebase.firestore.FirebaseFirestore;
import com.google.firebase.firestore.QueryDocumentSnapshot;
import com.google.firebase.firestore.QuerySnapshot;
import com.google.firebase.firestore.SetOptions;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.Map;

public class MainActivity extends AppCompatActivity {

    ImageButton camera, gallery;
    ImageView imageView;
    TextView result, kcal, carb, protein, fat;
    int imageSize = 32;
    FirebaseFirestore db = FirebaseFirestore.getInstance();
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        camera = findViewById(R.id.ibm_camera);
        gallery = findViewById(R.id.ibm_upload);
        kcal = findViewById(R.id.txvKcal);
        carb =  findViewById(R.id.txvCarbs);
        protein = findViewById(R.id.txvProtein);
        fat = findViewById(R.id.txvFat);
        result = findViewById(R.id.result);
        imageView = findViewById(R.id.imageView);

//        Map<String, Object> data = new HashMap<>();
//        data.put("name", "Dưa Chuột");
//        data.put("calo", 6);
//        data.put("carb", 2.1);
//        data.put("fat", 0);
//        data.put("protein", 3.8);
//        db.collection("fruits")
//                .add(data);

        camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 3);
                } else {
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });
        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent cameraIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(cameraIntent, 1);
            }
        });
    }
    public void classifyImage(Bitmap image){
        try {
            ModelFruits2 model = ModelFruits2.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 32, 32, 3}, DataType.FLOAT32);

            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int pixel = 0;
            //iterate over each pixel and extract R, G, and B values. Add those values individually to the byte buffer.
            for(int i = 0; i < imageSize; i ++){
                for(int j = 0; j < imageSize; j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 1));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 1));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 1));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            ModelFruits2.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            // find the index of the class with the biggest confidence.
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            String[] fruits = {
                    "Apple Braeburn", "Apple Crimson Snow", "Apple Golden 1", "Apple Golden 2", "Apple Golden 3",
                    "Apple Granny Smith", "Apple Pink Lady", "Apple Red 1", "Apple Red 2", "Apple Red 3",
                    "Apple Red Delicious", "Apple Red Yellow 1", "Apple Red Yellow 2", "Apricot", "Avocado",
                    "Avocado ripe", "Banana", "Banana Lady Finger", "Banana Red", "Beetroot", "Blueberry",
                    "Cactus fruit", "Cantaloupe 1", "Cantaloupe 2", "Carambula", "Cauliflower", "Cherry 1",
                    "Cherry 2", "Cherry Rainier", "Cherry Wax Black", "Cherry Wax Red", "Cherry Wax Yellow",
                    "Chestnut", "Clementine", "Cocos", "Corn", "Corn Husk", "Cucumber Ripe", "Cucumber Ripe 2",
                    "Dates", "Eggplant", "Fig", "Ginger Root", "Granadilla", "Grape Blue", "Grape Pink",
                    "Grape White", "Grape White 2", "Grape White 3", "Grape White 4", "Grapefruit Pink",
                    "Grapefruit White", "Guava", "Hazelnut", "Huckleberry", "Kaki", "Kiwi", "Kohlrabi",
                    "Kumquats", "Lemon", "Lemon Meyer", "Limes", "Lychee", "Mandarine", "Mango",
                    "Mango Red", "Mangostan", "Maracuja", "Melon Piel de Sapo", "Mulberry", "Nectarine",
                    "Nectarine Flat", "Nut Forest", "Nut Pecan", "Onion Red", "Onion Red Peeled", "Onion White",
                    "Orange", "Papaya", "Passion Fruit", "Peach", "Peach 2", "Peach Flat", "Pear",
                    "Pear 2", "Pear Abate", "Pear Forelle", "Pear Kaiser", "Pear Monster", "Pear Red",
                    "Pear Stone", "Pear Williams", "Pepino", "Pepper Green", "Pepper Orange", "Pepper Red",
                    "Pepper Yellow", "Physalis", "Physalis with Husk", "Pineapple", "Pineapple Mini",
                    "Pitahaya Red", "Plum", "Plum 2", "Plum 3", "Pomegranate", "Pomelo Sweetie", "Potato Red",
                    "Potato Red Washed", "Potato Sweet", "Potato White", "Quince", "Rambutan", "Raspberry",
                    "Redcurrant", "Salak", "Strawberry", "Strawberry Wedge", "Tamarillo", "Tangelo",
                    "Tomato 1", "Tomato 2", "Tomato 3", "Tomato 4", "Tomato Cherry Red", "Tomato Heart",
                    "Tomato Maroon", "Tomato Yellow", "Tomato not Ripened", "Walnut", "Watermelon"
            };
            String[] classes = {
                    "Quả Táo", "Quả Táo", "Quả Táo", "Quả Táo", "Quả Táo",
                    "Quả Táo", "Quả Táo", "Quả Táo", "Quả Táo", "Quả Táo",
                    "Quả Táo", "Quả Táo", "Quả Táo", "Quả Mơ", "Quả Bơ",
                    "Quả Bơ", "Quả Chuối", "Quả Chuối", "Quả Chuối", "Củ Cải Đường", "Quả Việt Quất",
                    "Quả Cactus", "Dưa Lưới", "Dưa Lưới", "Quả Khế", "Bông Cải Xanh", "Quả Anh Đào",
                    "Quả Anh Đào", "Quả Anh Đào", "Quả Anh Đào", "Quả Anh Đào", "Quả Anh Đào",
                    "Quả Hạt Dẻ", "Quả Clementine", "Quả Dừa", "Ngô", "Ngô", "Dưa Chuột", "Dưa Chuột",
                    "Quả Chà là", "Quả Cà Chua", "Quả Vối", "Gừng", "Quả Sầu Riêng", "Quả Nho",
                    "Quả Nho", "Quả Nho", "Quả Nho", "Quả Nho", "Quả Nho",
                    "Quả Bưởi", "Quả Bưởi", "Quả Ổi", "Quả Hạnh Nhân", "Quả Huckleberry", "Quả Kaki",
                    "Quả Kiwi", "Cải Xanh", "Quả Quất", "Quả Chanh", "Quả Chanh", "Quả Chanh",
                    "Quả Mãng Cầu", "Quả Bưởi", "Quả Bưởi", "Quả Bưởi", "Quả Bưởi", "Quả Bưởi",
                    "Quả Bưởi", "Quả Bưởi", "Quả Ổi", "Quả Lê", "Quả Lê", "Quả Lê",
                    "Quả Lê", "Quả Lê", "Quả Lê", "Quả Lê", "Quả Lê",
                    "Quả Lê", "Quả Pepino", "Quả Ớt", "Quả Ớt", "Quả Ớt", "Quả Ớt",
                    "Quả Physalis", "Quả Physalis", "Quả Dứa", "Quả Dứa", "Quả Thanh Long", "Quả Mận",
                    "Quả Mận", "Quả Mận", "Quả Lựu", "Quả Bưởi", "Khoai Tây", "Khoai Tây",
                    "Khoai Tây", "Khoai Tây", "Quả Mơ", "Quả Rambutan", "Quả Mâm Xôi",
                    "Quả Việt Quất", "Quả Dâu", "Quả Salak", "Quả Dâu", "Quả Dâu",
                    "Quả Tamarillo", "Quả Tangelo", "Quả Cà Chua", "Quả Cà Chua", "Quả Cà Chua", "Quả Cà Chua",
                    "Quả Cà Chua", "Quả Cà Chua", "Quả Cà Chua", "Quả Cà Chua", "Quả Cà Chua", "Hạt Điều", "Dưa Hấu"};

            result.setText(classes[maxPos]);

            db.collection("fruits")
                    .whereEqualTo("name", classes[maxPos])
                    .get()
                    .addOnCompleteListener(new OnCompleteListener<QuerySnapshot>() {
                        @Override
                        public void onComplete(@NonNull Task<QuerySnapshot> task) {
                            if (task.isSuccessful()) {
                                for (QueryDocumentSnapshot document : task.getResult()) {
                                    Ingredient ingredient = document.toObject(Ingredient.class);
                                    kcal.setText(String.valueOf(ingredient.getCalo()));
                                    carb.setText(String.valueOf(ingredient.getCarb()));
                                    fat.setText(String.valueOf(ingredient.getFat()));
                                    protein.setText(String.valueOf(ingredient.getProtein()));
                                }
                            } else {
                                Log.e(TAG, "Error getting documents: ", task.getException());
                            }
                        }
                    });



            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if(resultCode == RESULT_OK){
            if(requestCode == 3){
                Bitmap image = (Bitmap) data.getExtras().get("data");
                int dimension = Math.min(image.getWidth(), image.getHeight());
                image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
                imageView.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                classifyImage(image);
            }else{
                Uri dat = data.getData();
                Bitmap image = null;
                try {
                    image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dat);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                imageView.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                classifyImage(image);
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}
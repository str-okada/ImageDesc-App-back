const express = require('express');
const multer = require('multer');
const bodyParser = require('body-parser');
const cors = require('cors');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const mobilenet = require('@tensorflow-models/mobilenet');

const app = express();
app.use(bodyParser.json());
app.use(cors());

const PORT = 3080;

// 画像ファイルをアップロードするためのmulter設定
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/');
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + '-' + file.originalname);
  }
});

// const upload = multer({ storage: storage }).single('file');
const upload = multer({ storage });


// 画像分類APIのエンドポイント
app.post("/image", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ message: 'Could not upload the file' });
    }
    // 画像分類モデルのロード
    const model = await mobilenet.load();

    // 画像をTensorに変換
    const imagePath = req.file.path;
    const imageBuffer = fs.readFileSync(imagePath);
    const decodedImage = tf.node.decodeImage(imageBuffer);
    const reshapedImage = tf.image.resizeBilinear(decodedImage, [224, 224]);
    const scaledImage = reshapedImage.toFloat().div(tf.scalar(255)).expandDims();

    // モデルを使った処理を記述する
    const predictions = await model.classify(scaledImage);
    const predictedClass = predictions[0].className;

    // 結果をJSON形式で返す
    return res.json({ class: predictedClass });
  } catch (err) {
    return console.log(err)
  }
})

// app.post('/img', async (req, res, next) => {
//   try {
//     await upload(req, res, function (err) {
//       if (err instanceof multer.MulterError) {
//         // A Multer error occurred when uploading.
//         console.log(err);
//         res.status(400).json({ message: 'Error occurred when uploading' });
//       } else if (err) {
//         // An unknown error occurred when uploading.
//         console.log(err);
//         res.status(500).json({ message: 'Error occurred when uploading' });
//       }
//     });
//     if (!req.file) {
//       return res.status(400).json({ message: 'Could not upload the file' });
//     }

//     // 画像分類モデルのロード
//     const model = await mobilenet.load();

//     // 画像をTensorに変換
//     const imagePath = req.file.path;
//     const imageBuffer = fs.readFileSync(imagePath);
//     const decodedImage = tf.node.decodeImage(imageBuffer);
//     const reshapedImage = tf.image.resizeBilinear(decodedImage, [224, 224]);
//     const scaledImage = reshapedImage.toFloat().div(tf.scalar(255)).expandDims();

//     // モデルを使った処理を記述する
//     const predictions = await model.classify(scaledImage);
//     const predictedClass = predictions[0].className;

//     // 結果をJSON形式で返す
//     res.json({ class: predictedClass });
//   } catch (err) {
//     console.log(err);
//     res.status(500).json({ message: 'error occurred' });
//   }
// });

app.listen(PORT, () => {
  console.log(`http://localhost:${PORT}  // Start server`);
});

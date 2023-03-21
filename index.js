const express = require('express');
const multer = require('multer');
const bodyParser = require('body-parser');
const cors = require('cors');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const mobilenet = require('@tensorflow-models/mobilenet');
require('dotenv').config();

const app = express();
app.use(bodyParser.json());
app.use(cors());

// const PORT = 3080;


// open ai
const { Configuration, OpenAIApi } = require("openai");

const configuration = new Configuration({
  organization: "org-j0WGaD1mkube8Sv2HvAkgAYD",
  apiKey: process.env.APIKEY
});
const openai = new OpenAIApi(configuration);

app.post('/', async (req, res) => {
  const { message } = req.body;
  const response = await openai.createCompletion({
    model: "text-davinci-003",
    prompt: `${message}`,
    max_tokens: 100,
    temperature: 0.5,
  })
  res.json({
    message: response.data.choices[0].text,
  })
})

// setting for multer to upload the image
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/');
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + '-' + file.originalname);
  }
});

const upload = multer({ storage });


// end-point for API ti upload the image
app.post("/image", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ message: 'Could not upload the file' });
    }
    // loading the image
    const model = await mobilenet.load();

    // Convert image to Tensor
    const imagePath = req.file.path;
    const imageBuffer = fs.readFileSync(imagePath);
    const decodedImage = tf.node.decodeImage(imageBuffer);
    const reshapedImage = tf.image.resizeBilinear(decodedImage, [224, 224]);
    const scaledImage = reshapedImage.toFloat().div(tf.scalar(255)).expandDims();

    // const reshapedTensor = tf.reshape(decodedImage, [1, 224, 224, 3]);
    // const resizedTensor = tf.image.resizeBilinear(reshapedTensor, [newHeight, newWidth]);


    // モデルを使った処理
    const predictions = await model.classify(scaledImage);
    const predictedClass = predictions[0].className;

    // return the result in JSON format
    return res.json({ class: predictedClass });
  } catch (err) {
    return console.log(err)
  }
})

app.listen(process.env.PORT || 3080, () => {
  console.log(`http://localhost:3080  // Start server`);
});


// process.env.PORT


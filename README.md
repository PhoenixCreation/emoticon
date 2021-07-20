<h1 align="center">Emoticon 🙎➡️😎</h1>
<p align="center">Transfer yourself into live emoji with power of machine learning.</p>

#

## Index

- [Install](#Install)
- [Getting started](#Getting-Started)
  - [Customization](#Customization)
  - [Usage](#Usage)
- [Details](#Details)
- [Limitations](#Limitations)
- [Todo](#Todo)

#

## Install

1. Clone the repo and `cd` into it.

- (Optional)(`For Developers`)

  - Create a virtual environment and then go to step 2.
  - Run the following command to create a virtual environment.

    - For Windows:-

    ```
    python -m venv env
    ```

    - For Linux/MacOS:-

    ```
    python3 -m venv env
    ```

  - Run the following command to activate the virtual environment.

    ```
    source env/bin/activate
    ```

2. Install the dependencies using `pip`.

```
pip install -r requirements.txt
```

3. Run index.py file.

   - For Windows:-

   ```
   python index.py
   ```

   - For Linux/MacOS:-

   ```
   python3 index.py
   ```

## Getting Started

### Customization

- You can use the `any camera attached to your system` by changing the index of VideoCapture function. If you have 2 cameras then 0 will represent the default camera and other camera will have another index.(sometimes it might be not linear, sometime even for 2 camera setup, second camera's index might be 2 or 3, specially you are using virtual camera)

- Giving the VideoCapture function `path to a video` will run the program on that video. This is done mostly for `testing` purposes.

- Change the emoji as per your need in `emojis` folder. `DON'T change the name. Make sure to rename your emoji photos as already given.`

### Usage

- Create `sequence` of different mood showing emoji.

- Use the `emoticon screen as a webcam for your meetings` if you just want to give reactions in meeting.(That's what 90% people do in online meetings)

## Details

- This uses `DeepFace` library to predict the emotion. You can see more details [here.](https://github.com/serengil/deepface)

- It has quite good prediction rate. `80%-90%` time it predicts the right emotion unless you are not good at expressing them.

## Limitations

There are some limitations as of now. Might get eliminated in future if possible.

- Not good performance in `low light`. You will need quite good quality webcam. One solution is to use your phone's camera as a webcam.

- Does not interact with `side faces`. It needs straight looking face.

- Emotion flatulates very much sometimes. It can go from sad to happy in one frame which is not practical.

## Todo

[ ] Fix the flatulatation problem with counter and average emotion by `keeping track of past emotions`.

[ ] Implement different versions of emoji for different `genders`. Can be followed up by other parameters like `eye color, skin color etc`.

[ ] `(Should be done parallely)` Transfer the emojis to `SVG` and if possible then animatable SVGs for better customization and cool effects. If not then `GIFs` are also an reliable option.

#

## Contributors

- Het S. Patel - [@PhoenixCreation](https://github.com/PhoenixCreation)

#

### Give a 🌟 to this repo if you enjoyed being transferred to emojis.

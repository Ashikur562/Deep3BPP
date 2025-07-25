{
  "cells": [
    {
      "cell_type": "code",
      "source": [
        "!python --version"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8womJLE_Kt-K",
        "outputId": "123b9fbe-f6e5-4657-bcb4-47c78d056361"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Python 3.10.12\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "LiP_oCmVagEe"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sb\n",
        "from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, cohen_kappa_score\n",
        "from sklearn.metrics import matthews_corrcoef\n",
        "from sklearn.model_selection import KFold\n",
        "from sklearn.neural_network import MLPClassifier\n",
        "from sklearn.model_selection import cross_val_score, cross_val_predict\n",
        "from keras.models import Sequential\n",
        "from keras.layers import Dense, Conv1D, MaxPool1D, Flatten, Dropout,LSTM"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "lEscnva-bol5"
      },
      "source": [
        "**Train-Test Split**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "B-wLNLy6brtU"
      },
      "outputs": [],
      "source": [
        "# from sklearn.model_selection import train_test_split\n",
        "# xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.2, random_state = 1)\n",
        "# # xtrain = xtrain.to_numpy()\n",
        "# # xtest = xtest.to_numpy()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "DdbnMkhvbzYC"
      },
      "source": [
        "# **CNN+LSTM(LSA)**"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "sSnuM29foevd"
      },
      "source": [
        "**Imbalanced**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "jSWHfbOUbk6C"
      },
      "outputs": [],
      "source": [
        "# df = pd.read_csv('/content/LSA_DPI_Dataset.csv')\n",
        "# columns = df.columns.tolist()\n",
        "# # Filter the columns to remove data we do not want\n",
        "# columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# # Store the variable we are predicting\n",
        "# target = \"Target\"\n",
        "# X = df[columns]\n",
        "# Y = df[target]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "2U4Ho--qq22x"
      },
      "outputs": [],
      "source": [
        "df = pd.read_csv('/content/LSA_DPI_Dataset.csv')\n",
        "columns = df.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df[columns]\n",
        "Y = df[target]\n",
        "\n",
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)\n",
        "\n",
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]\n",
        "\n",
        "# def CNN_LSTM(optimizer='sgd', kernel_size=3, filters1=32, filters2=64, pool_size=2, lstm_units=64, dense_units=1, dropout_rate=0.5, reg_param=0.01):\n",
        "#     model = Sequential()\n",
        "#     model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))\n",
        "#     model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))\n",
        "#     # cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))\n",
        "#     model.add(MaxPool1D(pool_size=4))\n",
        "#     model.add(LSTM(256, activation='relu'))\n",
        "#     model.add(Flatten())\n",
        "#     model.add(Dense(64, activation='relu'))\n",
        "#     model.add(Dense(1, activation='sigmoid'))\n",
        "#     model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])\n",
        "\n",
        "#     return model\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "qr8pym1T64yn"
      },
      "outputs": [],
      "source": [
        "# fold_metrics = []"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "6D0sM-kPrGQx"
      },
      "outputs": [],
      "source": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "4G19dIDNpx73"
      },
      "outputs": [],
      "source": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "KXzdvXItpzu9"
      },
      "outputs": [],
      "source": [
        "\n",
        "model = Sequential()\n",
        "model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))\n",
        "model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))\n",
        "# model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))\n",
        "model.add(MaxPool1D(pool_size=4))\n",
        "model.add(LSTM(256, activation='relu'))\n",
        "model.add(Flatten())\n",
        "model.add(Dense(64, activation='relu'))\n",
        "model.add(Dense(1, activation='sigmoid'))\n",
        "model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "model.summary()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "n9OmZf1HF9ZM",
        "outputId": "a407c814-ea03-4326-9a7c-2de3107767c6"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model: \"sequential_1\"\n",
            "_________________________________________________________________\n",
            " Layer (type)                Output Shape              Param #   \n",
            "=================================================================\n",
            " conv1d (Conv1D)             (None, 18, 64)            256       \n",
            "                                                                 \n",
            " conv1d_1 (Conv1D)           (None, 16, 128)           24704     \n",
            "                                                                 \n",
            " max_pooling1d (MaxPooling1  (None, 4, 128)            0         \n",
            " D)                                                              \n",
            "                                                                 \n",
            " lstm (LSTM)                 (None, 256)               394240    \n",
            "                                                                 \n",
            " flatten (Flatten)           (None, 256)               0         \n",
            "                                                                 \n",
            " dense (Dense)               (None, 64)                16448     \n",
            "                                                                 \n",
            " dense_1 (Dense)             (None, 1)                 65        \n",
            "                                                                 \n",
            "=================================================================\n",
            "Total params: 435713 (1.66 MB)\n",
            "Trainable params: 435713 (1.66 MB)\n",
            "Non-trainable params: 0 (0.00 Byte)\n",
            "_________________________________________________________________\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "b22I6mjUQLH2",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "38dab506-ec0a-4264-c2af-c67352203376"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "26/26 [==============================] - 6s 62ms/step - loss: 0.6930 - accuracy: 0.5103\n",
            "Epoch 2/75\n",
            "26/26 [==============================] - 2s 82ms/step - loss: 0.6929 - accuracy: 0.5152\n",
            "Epoch 3/75\n",
            "26/26 [==============================] - 2s 73ms/step - loss: 0.6929 - accuracy: 0.5152\n",
            "Epoch 4/75\n",
            "26/26 [==============================] - 2s 70ms/step - loss: 0.6928 - accuracy: 0.5152\n",
            "Epoch 5/75\n",
            "26/26 [==============================] - 2s 64ms/step - loss: 0.6929 - accuracy: 0.5152\n",
            "Epoch 6/75\n",
            "26/26 [==============================] - 2s 75ms/step - loss: 0.6927 - accuracy: 0.5152\n",
            "Epoch 7/75\n",
            "26/26 [==============================] - 2s 62ms/step - loss: 0.6928 - accuracy: 0.5170\n",
            "Epoch 8/75\n",
            "26/26 [==============================] - 2s 61ms/step - loss: 0.6928 - accuracy: 0.5176\n",
            "Epoch 9/75\n",
            "26/26 [==============================] - 2s 60ms/step - loss: 0.6926 - accuracy: 0.5121\n",
            "Epoch 10/75\n",
            "26/26 [==============================] - 2s 71ms/step - loss: 0.6919 - accuracy: 0.5170\n",
            "Epoch 11/75\n",
            "26/26 [==============================] - 2s 72ms/step - loss: 0.6915 - accuracy: 0.5170\n",
            "Epoch 12/75\n",
            "26/26 [==============================] - 2s 76ms/step - loss: 0.6914 - accuracy: 0.5170\n",
            "Epoch 13/75\n",
            "26/26 [==============================] - 2s 73ms/step - loss: 0.6914 - accuracy: 0.5170\n",
            "Epoch 14/75\n",
            "26/26 [==============================] - 2s 76ms/step - loss: 0.6914 - accuracy: 0.5170\n",
            "Epoch 15/75\n",
            "26/26 [==============================] - 2s 71ms/step - loss: 0.6914 - accuracy: 0.5170\n",
            "Epoch 16/75\n",
            "26/26 [==============================] - 2s 57ms/step - loss: 0.6914 - accuracy: 0.5170\n",
            "Epoch 17/75\n",
            "26/26 [==============================] - 2s 68ms/step - loss: 0.6915 - accuracy: 0.5170\n",
            "Epoch 18/75\n",
            "26/26 [==============================] - 2s 78ms/step - loss: 0.6914 - accuracy: 0.5170\n",
            "Epoch 19/75\n",
            "26/26 [==============================] - 3s 97ms/step - loss: 0.6914 - accuracy: 0.5170\n",
            "Epoch 20/75\n",
            "26/26 [==============================] - 2s 94ms/step - loss: 0.6914 - accuracy: 0.5170\n",
            "Epoch 21/75\n",
            "26/26 [==============================] - 3s 96ms/step - loss: 0.6915 - accuracy: 0.5170\n",
            "Epoch 22/75\n",
            "26/26 [==============================] - 2s 90ms/step - loss: 0.6914 - accuracy: 0.5170\n",
            "Epoch 23/75\n",
            "26/26 [==============================] - 2s 74ms/step - loss: 0.6914 - accuracy: 0.5170\n",
            "Epoch 24/75\n",
            "26/26 [==============================] - 2s 60ms/step - loss: 0.6914 - accuracy: 0.5170\n",
            "Epoch 25/75\n",
            "26/26 [==============================] - 2s 59ms/step - loss: 0.6916 - accuracy: 0.5170\n",
            "Epoch 26/75\n",
            "26/26 [==============================] - 2s 65ms/step - loss: 0.6914 - accuracy: 0.5170\n",
            "Epoch 27/75\n",
            "26/26 [==============================] - 1s 57ms/step - loss: 0.6914 - accuracy: 0.5170\n",
            "Epoch 28/75\n",
            "26/26 [==============================] - 1s 49ms/step - loss: 0.6914 - accuracy: 0.5170\n",
            "Epoch 29/75\n",
            "26/26 [==============================] - 1s 51ms/step - loss: 0.6914 - accuracy: 0.5170\n",
            "Epoch 30/75\n",
            "26/26 [==============================] - 2s 74ms/step - loss: 0.6915 - accuracy: 0.5170\n",
            "Epoch 31/75\n",
            "26/26 [==============================] - 2s 73ms/step - loss: 0.6915 - accuracy: 0.5170\n",
            "Epoch 32/75\n",
            "26/26 [==============================] - 2s 76ms/step - loss: 0.6913 - accuracy: 0.5170\n",
            "Epoch 33/75\n",
            "26/26 [==============================] - 1s 50ms/step - loss: 0.6915 - accuracy: 0.5188\n",
            "Epoch 34/75\n",
            "26/26 [==============================] - 2s 59ms/step - loss: 0.6913 - accuracy: 0.5170\n",
            "Epoch 35/75\n",
            "26/26 [==============================] - 1s 53ms/step - loss: 0.6912 - accuracy: 0.5176\n",
            "Epoch 36/75\n",
            "26/26 [==============================] - 1s 41ms/step - loss: 0.6918 - accuracy: 0.5103\n",
            "Epoch 37/75\n",
            "26/26 [==============================] - 1s 35ms/step - loss: 0.6913 - accuracy: 0.5170\n",
            "Epoch 38/75\n",
            "26/26 [==============================] - 1s 39ms/step - loss: 0.6917 - accuracy: 0.5067\n",
            "Epoch 39/75\n",
            "26/26 [==============================] - 1s 39ms/step - loss: 0.6908 - accuracy: 0.5170\n",
            "Epoch 40/75\n",
            "26/26 [==============================] - 1s 40ms/step - loss: 0.6908 - accuracy: 0.5242\n",
            "Epoch 41/75\n",
            "26/26 [==============================] - 1s 54ms/step - loss: 0.6917 - accuracy: 0.5194\n",
            "Epoch 42/75\n",
            "26/26 [==============================] - 1s 57ms/step - loss: 0.6907 - accuracy: 0.5200\n",
            "Epoch 43/75\n",
            "26/26 [==============================] - 2s 59ms/step - loss: 0.6902 - accuracy: 0.5291\n",
            "Epoch 44/75\n",
            "26/26 [==============================] - 1s 41ms/step - loss: 0.6896 - accuracy: 0.5242\n",
            "Epoch 45/75\n",
            "26/26 [==============================] - 1s 40ms/step - loss: 0.6900 - accuracy: 0.5139\n",
            "Epoch 46/75\n",
            "26/26 [==============================] - 1s 40ms/step - loss: 0.6927 - accuracy: 0.5158\n",
            "Epoch 47/75\n",
            "26/26 [==============================] - 1s 39ms/step - loss: 0.6904 - accuracy: 0.5230\n",
            "Epoch 48/75\n",
            "26/26 [==============================] - 1s 35ms/step - loss: 0.6902 - accuracy: 0.5327\n",
            "Epoch 49/75\n",
            "26/26 [==============================] - 1s 39ms/step - loss: 0.6884 - accuracy: 0.5418\n",
            "Epoch 50/75\n",
            "26/26 [==============================] - 1s 34ms/step - loss: 0.6869 - accuracy: 0.5448\n",
            "Epoch 51/75\n",
            "26/26 [==============================] - 1s 34ms/step - loss: 0.6888 - accuracy: 0.5285\n",
            "Epoch 52/75\n",
            "26/26 [==============================] - 1s 33ms/step - loss: 0.6878 - accuracy: 0.5267\n",
            "Epoch 53/75\n",
            "26/26 [==============================] - 1s 38ms/step - loss: 0.6864 - accuracy: 0.5364\n",
            "Epoch 54/75\n",
            "26/26 [==============================] - 1s 52ms/step - loss: 0.6875 - accuracy: 0.5321\n",
            "Epoch 55/75\n",
            "26/26 [==============================] - 1s 58ms/step - loss: 0.6878 - accuracy: 0.5364\n",
            "Epoch 56/75\n",
            "26/26 [==============================] - 1s 55ms/step - loss: 0.6864 - accuracy: 0.5255\n",
            "Epoch 57/75\n",
            "26/26 [==============================] - 1s 44ms/step - loss: 0.6848 - accuracy: 0.5376\n",
            "Epoch 58/75\n",
            "26/26 [==============================] - 1s 40ms/step - loss: 0.6846 - accuracy: 0.5394\n",
            "Epoch 59/75\n",
            "26/26 [==============================] - 1s 33ms/step - loss: 0.6850 - accuracy: 0.5455\n",
            "Epoch 60/75\n",
            "26/26 [==============================] - 1s 41ms/step - loss: 0.6862 - accuracy: 0.5418\n",
            "Epoch 61/75\n",
            "26/26 [==============================] - 1s 33ms/step - loss: 0.6847 - accuracy: 0.5333\n",
            "Epoch 62/75\n",
            "26/26 [==============================] - 1s 39ms/step - loss: 0.6829 - accuracy: 0.5558\n",
            "Epoch 63/75\n",
            "26/26 [==============================] - 1s 38ms/step - loss: 0.6804 - accuracy: 0.5418\n",
            "Epoch 64/75\n",
            "26/26 [==============================] - 1s 39ms/step - loss: 0.6790 - accuracy: 0.5558\n",
            "Epoch 65/75\n",
            "26/26 [==============================] - 1s 33ms/step - loss: 0.6742 - accuracy: 0.5758\n",
            "Epoch 66/75\n",
            "26/26 [==============================] - 1s 34ms/step - loss: 0.6680 - accuracy: 0.5818\n",
            "Epoch 67/75\n",
            "26/26 [==============================] - 1s 46ms/step - loss: 0.6580 - accuracy: 0.5994\n",
            "Epoch 68/75\n",
            "26/26 [==============================] - 1s 55ms/step - loss: 0.6348 - accuracy: 0.6345\n",
            "Epoch 69/75\n",
            "26/26 [==============================] - 2s 58ms/step - loss: 0.6115 - accuracy: 0.6491\n",
            "Epoch 70/75\n",
            "26/26 [==============================] - 1s 49ms/step - loss: 0.5664 - accuracy: 0.6939\n",
            "Epoch 71/75\n",
            "26/26 [==============================] - 1s 33ms/step - loss: 0.5245 - accuracy: 0.7309\n",
            "Epoch 72/75\n",
            "26/26 [==============================] - 1s 35ms/step - loss: 0.5079 - accuracy: 0.7436\n",
            "Epoch 73/75\n",
            "26/26 [==============================] - 1s 34ms/step - loss: 0.4797 - accuracy: 0.7552\n",
            "Epoch 74/75\n",
            "26/26 [==============================] - 1s 33ms/step - loss: 0.4975 - accuracy: 0.7467\n",
            "Epoch 75/75\n",
            "26/26 [==============================] - 1s 34ms/step - loss: 0.4702 - accuracy: 0.7685\n"
          ]
        }
      ],
      "source": [
        "history = model.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "FaU2lwo65PTH",
        "outputId": "531b3d87-3090-4e66-c5ee-de0d56924cff"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "13/13 [==============================] - 0s 12ms/step\n"
          ]
        }
      ],
      "source": [
        "pred = model.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "5deYSvw8IAdD",
        "outputId": "a33b3375-38e0-46fb-c615-3563e3c71303"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.\n",
            "  _warn_prf(average, modifier, msg_start, len(result))\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.5169902912621359, 0.0, 0.0, 0.0, 0.0)"
            ]
          },
          "metadata": {},
          "execution_count": 21
        }
      ],
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import roc_curve, roc_auc_score\n",
        "y_pred_proba = y_pred_classes.flatten()\n",
        "\n",
        "# Calculate ROC curve and AUC score\n",
        "fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)\n",
        "roc_auc = roc_auc_score(y_val, y_pred_proba)"
      ],
      "metadata": {
        "id": "wAYON3ggd3zb"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Plot ROC curve\n",
        "plt.figure(figsize=(8, 6))\n",
        "plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')\n",
        "plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')\n",
        "plt.xlabel('False Positive Rate')\n",
        "plt.ylabel('True Positive Rate')\n",
        "plt.title('ROC Curve')\n",
        "plt.legend(loc='lower right')\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 564
        },
        "id": "33jgY0O_ebMy",
        "outputId": "e34f28e8-6222-4518-c7ee-a7800aab9aa7"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 800x600 with 1 Axes>"
            ],
            "image/png": "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uLq6GmFhYfe91qtXL6NixYr25126dDECAgKMH374wTh16pTx22+/GVWrVjWqVq1q2Gw2+3aTJ082LBaL8eabbxobN240zpw5Y2zZssXo0qWL0aNHjxT7Lu+8845RoEAB49dffzX++OMPo1q1aka1atXibVOsWDFj6dKlhmHcvYJDz549je3btxunT5821q1bZ5QvX94oUqSIER0dbd+nfv36Rrly5YwdO3YYW7ZsMYoUKWK0bNnygTmS62oGppdZwzCMr776yihQoIDh6upqVK5c2fj999/tr9WqVcto3759vO0XLVpkFC1a1HB1dTWeffZZY+XKlYn+LJVZERERczlimX311VeNBg0aJPjajh07DMDYv3+/YRh3v9+AAQOM4sWLGx4eHkbBggWNLl26GFeuXLlv37Vr1xr16tUzfH19DXd3d6N48eJGz549jQsXLqTYd4mKijLee+89w9fX1/D09DSaNGliXLx4Md42gDFr1izDMAzj9u3bRt26dY2cOXMaLi4uxlNPPWW89dZb9onHe65du2a0bNnSyJIli+Hl5WV07NjRuHnz5kNzJEeZtfwbOMOIiIjA29ub8PBw+7oWERERST3R0dGcPn2aggUL4u7ubnYcMcnD/jtISl8z/Q5gIiIiIiKPS2VWRERERByWyqyIiIiIOCyVWRERERFxWCqzIiIiYooMdg66/D/JdfxVZkVERCRVubi4AHD79m2Tk4iZYmJiAHB2dn6i9zH1drYiIiKS8Tg7O+Pj42O//amnpycWi8XkVJKabDYbV65cwdPTk0yZnqyOqsyKiIhIqvPz8wOwF1rJeJycnChQoMAT/0VGZVZERERSncViIU+ePOTKlYvY2Fiz44gJXF1dcXJ68hWvKrMiIiJiGmdn5ydeMykZm04AExERERGHpTIrIiIiIg5LZVZEREREHFaGWzN77wK9ERERJicRERERkYTc62mJubFChiuzN2/eBMDf39/kJCIiIiLyMDdv3sTb2/uh21iMDHYvOZvNxoULF8iaNWuqXKA5IiICf39/zp07h5eXV4p/niQ/HUPHp2Po+HQMHZuOn+NL7WNoGAY3b94kb968j7x8V4abmXVyciJ//vyp/rleXl76DezgdAwdn46h49MxdGw6fo4vNY/ho2Zk79EJYCIiIiLisFRmRURERMRhqcymMDc3NwYMGICbm5vZ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ozg5OSW4j5eXF3nz5mXr1q3UqlXLPr5161YqV64cb7ugoCCCgoJ4/fXXqV+/PtevXydbtmzx3u/e+lSr1frQPO7u7jRt2pR58+Zx4sQJihUrRvny5e2vly9fnmPHjiX5e/5/n3/+OXXq1OHdd9+1f8/q1avz3nvv2bf5/zOrrq6u9+UvX748ISEh5MqVCy8vryfKJCLph04AE5F07+mnnyY2NpavvvqKU6dOMWfOHKZOnfrA7aOionj//ffZuHEjf//9N1u3bmXXrl325QOffvop27Zt4/3332ffvn389ddf/PDDD0k+Aex/ffLJJ4wcOZKQkBCOHTtG79692bdvH927dwdg3LhxLFiwgKNHj3L8+HEWL16Mn59fgjd6yJUrFx4eHqxevZpLly4RHh7+wM9t3bo1K1euZObMmfYTv+7p378/s2fPZuDAgRw6dIgjR46wcOFCPv/88yR9t2rVqlG6dGmGDRsGQJEiRfjjjz9Ys2YNx48fp1+/fuzatSvePgEBARw4cIBjx45x9epVYmNjad26NTly5KBRo0Zs3ryZ06dPs3HjRj744AP++eefJGUSkfRDZVZE0r0yZcowbtw4Ro4cScmSJZk3b168y1r9f87Ozly7do127dpRtGhRmjdvzssvv8zAgQMBKF26NL/99hvHjx+nZs2alCtXjv79+5M3b97HzvjBBx/Qo0cPPv74Y0qVKsXq1atZsWIFRYoUAe4uURg1ahQVK1akUqVKnDlzhlWrVtlnmv9XpkyZ+PLLL/nmm2/ImzcvjRo1euDn1qlTh2zZsnHs2DFatWoV77V69erx008/8csvv1CpUiWqVq3K+PHjeeqpp5L8/T766COmT5/OuXPnePvtt2natClBQUFUqVKFa9euxZulBXjrrbcoVqwYFStWJGfOnGzduhVPT082bdpEgQIFaNq0KSVKlKBTp05ER0drplYkA7MYhmGYHUJERERE5HFoZlZEREREHJbKrIiIiIg4LJVZEREREXFYKrMiIiIi4rBUZkVERETEYanMioiIiIjDUpkVEREREYelMisiIiIiDktlVkREREQclsqsiIiIiDgslVkRERERcVj/B4A6wa/pcaUlAAAAAElFTkSuQmCC\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "6EJjG-fmJly0",
        "outputId": "f0af5737-60bd-432c-ec81-0a0d89488e39"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9942196531791907"
            ]
          },
          "metadata": {},
          "execution_count": 152
        }
      ],
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Test**"
      ],
      "metadata": {
        "id": "R8UQ3zHDePAf"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/LSA_TR.csv')\n",
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "4_IjGi-VeQ9D"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.3, random_state = 1)"
      ],
      "metadata": {
        "id": "wS13DDv7mB-M"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "sample_size = xtrain.shape[0] # number of samples in train set\n",
        "time_steps  = xtrain.shape[1] # number of features in train set\n",
        "input_dimension = 1               # each feature is represented by 1 number"
      ],
      "metadata": {
        "id": "mfrIlTTMetjM"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_data_reshaped = xtrain.values.reshape(sample_size,time_steps,input_dimension)\n",
        "n_timesteps = train_data_reshaped.shape[1]\n",
        "n_features  = train_data_reshaped.shape[2]"
      ],
      "metadata": {
        "id": "3jz8_1JJe4hi"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn = Sequential()"
      ],
      "metadata": {
        "id": "FSoc8twNe-yU"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))\n",
        "cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))\n",
        "# cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))"
      ],
      "metadata": {
        "id": "C19xuqwcfHJd"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(MaxPool1D(pool_size=4))"
      ],
      "metadata": {
        "id": "4ufXtA0KfQVj"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(LSTM(256, activation='relu'))"
      ],
      "metadata": {
        "id": "QdkK1dzMfUOc"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Flatten())"
      ],
      "metadata": {
        "id": "GBYcMbeOfYU1"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "8GyuPCQLfcQ_"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "nuNapn8IfghI"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.fit(xtrain, ytrain, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "UqZuMsLuflQE",
        "outputId": "53ae11d7-44e7-4086-d4a6-1c2c513ae803"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "36/36 [==============================] - 9s 135ms/step - loss: 0.5558 - accuracy: 0.8127\n",
            "Epoch 2/75\n",
            "36/36 [==============================] - 6s 181ms/step - loss: 0.4911 - accuracy: 0.8127\n",
            "Epoch 3/75\n",
            "36/36 [==============================] - 5s 130ms/step - loss: 0.4891 - accuracy: 0.8127\n",
            "Epoch 4/75\n",
            "36/36 [==============================] - 5s 148ms/step - loss: 0.4854 - accuracy: 0.8127\n",
            "Epoch 5/75\n",
            "36/36 [==============================] - 5s 144ms/step - loss: 0.4813 - accuracy: 0.8127\n",
            "Epoch 6/75\n",
            "36/36 [==============================] - 5s 127ms/step - loss: 0.4656 - accuracy: 0.8127\n",
            "Epoch 7/75\n",
            "36/36 [==============================] - 4s 111ms/step - loss: 0.4102 - accuracy: 0.8251\n",
            "Epoch 8/75\n",
            "36/36 [==============================] - 3s 93ms/step - loss: 0.3715 - accuracy: 0.8454\n",
            "Epoch 9/75\n",
            "36/36 [==============================] - 3s 85ms/step - loss: 0.3509 - accuracy: 0.8494\n",
            "Epoch 10/75\n",
            "36/36 [==============================] - 3s 86ms/step - loss: 0.3209 - accuracy: 0.8636\n",
            "Epoch 11/75\n",
            "36/36 [==============================] - 4s 120ms/step - loss: 0.2843 - accuracy: 0.8760\n",
            "Epoch 12/75\n",
            "36/36 [==============================] - 3s 86ms/step - loss: 0.2381 - accuracy: 0.8919\n",
            "Epoch 13/75\n",
            "36/36 [==============================] - 3s 85ms/step - loss: 0.2352 - accuracy: 0.8977\n",
            "Epoch 14/75\n",
            "36/36 [==============================] - 3s 87ms/step - loss: 0.1941 - accuracy: 0.9159\n",
            "Epoch 15/75\n",
            "36/36 [==============================] - 5s 139ms/step - loss: 0.1446 - accuracy: 0.9384\n",
            "Epoch 16/75\n",
            "36/36 [==============================] - 4s 103ms/step - loss: 0.1210 - accuracy: 0.9446\n",
            "Epoch 17/75\n",
            "36/36 [==============================] - 3s 85ms/step - loss: 0.1155 - accuracy: 0.9517\n",
            "Epoch 18/75\n",
            "36/36 [==============================] - 3s 84ms/step - loss: 0.1209 - accuracy: 0.9504\n",
            "Epoch 19/75\n",
            "36/36 [==============================] - 4s 120ms/step - loss: 0.0926 - accuracy: 0.9637\n",
            "Epoch 20/75\n",
            "36/36 [==============================] - 3s 86ms/step - loss: 0.0903 - accuracy: 0.9615\n",
            "Epoch 21/75\n",
            "36/36 [==============================] - 3s 86ms/step - loss: 0.0830 - accuracy: 0.9659\n",
            "Epoch 22/75\n",
            "36/36 [==============================] - 3s 87ms/step - loss: 0.0654 - accuracy: 0.9748\n",
            "Epoch 23/75\n",
            "36/36 [==============================] - 4s 118ms/step - loss: 0.0840 - accuracy: 0.9663\n",
            "Epoch 24/75\n",
            "36/36 [==============================] - 3s 85ms/step - loss: 0.2180 - accuracy: 0.9229\n",
            "Epoch 25/75\n",
            "36/36 [==============================] - 3s 84ms/step - loss: 0.0948 - accuracy: 0.9619\n",
            "Epoch 26/75\n",
            "36/36 [==============================] - 3s 96ms/step - loss: 0.0621 - accuracy: 0.9765\n",
            "Epoch 27/75\n",
            "36/36 [==============================] - 4s 106ms/step - loss: 0.0742 - accuracy: 0.9686\n",
            "Epoch 28/75\n",
            "36/36 [==============================] - 3s 85ms/step - loss: 0.0633 - accuracy: 0.9743\n",
            "Epoch 29/75\n",
            "36/36 [==============================] - 3s 85ms/step - loss: 0.0694 - accuracy: 0.9734\n",
            "Epoch 30/75\n",
            "36/36 [==============================] - 4s 106ms/step - loss: 0.0416 - accuracy: 0.9849\n",
            "Epoch 31/75\n",
            "36/36 [==============================] - 4s 98ms/step - loss: 0.0606 - accuracy: 0.9734\n",
            "Epoch 32/75\n",
            "36/36 [==============================] - 3s 86ms/step - loss: 0.3114 - accuracy: 0.9128\n",
            "Epoch 33/75\n",
            "36/36 [==============================] - 3s 84ms/step - loss: 0.2872 - accuracy: 0.8795\n",
            "Epoch 34/75\n",
            "36/36 [==============================] - 4s 114ms/step - loss: 0.1640 - accuracy: 0.9411\n",
            "Epoch 35/75\n",
            "36/36 [==============================] - 3s 88ms/step - loss: 0.1172 - accuracy: 0.9548\n",
            "Epoch 36/75\n",
            "36/36 [==============================] - 3s 85ms/step - loss: 0.1017 - accuracy: 0.9606\n",
            "Epoch 37/75\n",
            "36/36 [==============================] - 3s 86ms/step - loss: 0.0761 - accuracy: 0.9756\n",
            "Epoch 38/75\n",
            "36/36 [==============================] - 4s 120ms/step - loss: 0.0716 - accuracy: 0.9756\n",
            "Epoch 39/75\n",
            "36/36 [==============================] - 3s 86ms/step - loss: 0.0575 - accuracy: 0.9796\n",
            "Epoch 40/75\n",
            "36/36 [==============================] - 3s 86ms/step - loss: 0.0587 - accuracy: 0.9779\n",
            "Epoch 41/75\n",
            "36/36 [==============================] - 3s 86ms/step - loss: 0.0402 - accuracy: 0.9872\n",
            "Epoch 42/75\n",
            "36/36 [==============================] - 4s 121ms/step - loss: 0.0338 - accuracy: 0.9903\n",
            "Epoch 43/75\n",
            "36/36 [==============================] - 3s 85ms/step - loss: 0.0567 - accuracy: 0.9832\n",
            "Epoch 44/75\n",
            "36/36 [==============================] - 3s 84ms/step - loss: 0.0885 - accuracy: 0.9686\n",
            "Epoch 45/75\n",
            "36/36 [==============================] - 3s 84ms/step - loss: 0.0400 - accuracy: 0.9889\n",
            "Epoch 46/75\n",
            "36/36 [==============================] - 4s 118ms/step - loss: 0.0263 - accuracy: 0.9938\n",
            "Epoch 47/75\n",
            "36/36 [==============================] - 3s 85ms/step - loss: 0.0213 - accuracy: 0.9947\n",
            "Epoch 48/75\n",
            "36/36 [==============================] - 3s 85ms/step - loss: 0.0259 - accuracy: 0.9925\n",
            "Epoch 49/75\n",
            "36/36 [==============================] - 4s 98ms/step - loss: 0.0218 - accuracy: 0.9942\n",
            "Epoch 50/75\n",
            "36/36 [==============================] - 4s 107ms/step - loss: 0.0154 - accuracy: 0.9960\n",
            "Epoch 51/75\n",
            "36/36 [==============================] - 3s 87ms/step - loss: 0.0529 - accuracy: 0.9836\n",
            "Epoch 52/75\n",
            "36/36 [==============================] - 3s 85ms/step - loss: 0.0256 - accuracy: 0.9925\n",
            "Epoch 53/75\n",
            "36/36 [==============================] - 4s 109ms/step - loss: 0.0154 - accuracy: 0.9956\n",
            "Epoch 54/75\n",
            "36/36 [==============================] - 3s 95ms/step - loss: 0.0095 - accuracy: 0.9987\n",
            "Epoch 55/75\n",
            "36/36 [==============================] - 3s 85ms/step - loss: 0.0105 - accuracy: 0.9987\n",
            "Epoch 56/75\n",
            "36/36 [==============================] - 3s 85ms/step - loss: 0.0088 - accuracy: 0.9973\n",
            "Epoch 57/75\n",
            "36/36 [==============================] - 4s 118ms/step - loss: 0.0080 - accuracy: 0.9982\n",
            "Epoch 58/75\n",
            "36/36 [==============================] - 5s 125ms/step - loss: 0.0123 - accuracy: 0.9951\n",
            "Epoch 59/75\n",
            "36/36 [==============================] - 3s 86ms/step - loss: 0.0184 - accuracy: 0.9934\n",
            "Epoch 60/75\n",
            "36/36 [==============================] - 3s 87ms/step - loss: 0.0205 - accuracy: 0.9942\n",
            "Epoch 61/75\n",
            "36/36 [==============================] - 4s 120ms/step - loss: 0.0116 - accuracy: 0.9965\n",
            "Epoch 62/75\n",
            "36/36 [==============================] - 3s 87ms/step - loss: 0.0084 - accuracy: 0.9978\n",
            "Epoch 63/75\n",
            "36/36 [==============================] - 3s 85ms/step - loss: 0.0092 - accuracy: 0.9978\n",
            "Epoch 64/75\n",
            "36/36 [==============================] - 5s 132ms/step - loss: 0.0362 - accuracy: 0.9885\n",
            "Epoch 65/75\n",
            "36/36 [==============================] - 4s 108ms/step - loss: 0.0579 - accuracy: 0.9752\n",
            "Epoch 66/75\n",
            "36/36 [==============================] - 3s 85ms/step - loss: 0.0160 - accuracy: 0.9960\n",
            "Epoch 67/75\n",
            "36/36 [==============================] - 3s 86ms/step - loss: 0.0069 - accuracy: 0.9991\n",
            "Epoch 68/75\n",
            "36/36 [==============================] - 4s 109ms/step - loss: 0.0041 - accuracy: 0.9991\n",
            "Epoch 69/75\n",
            "36/36 [==============================] - 3s 95ms/step - loss: 0.0031 - accuracy: 0.9996\n",
            "Epoch 70/75\n",
            "36/36 [==============================] - 3s 85ms/step - loss: 0.0033 - accuracy: 0.9996\n",
            "Epoch 71/75\n",
            "36/36 [==============================] - 3s 86ms/step - loss: 0.0024 - accuracy: 0.9991\n",
            "Epoch 72/75\n",
            "36/36 [==============================] - 4s 120ms/step - loss: 0.0020 - accuracy: 0.9996\n",
            "Epoch 73/75\n",
            "36/36 [==============================] - 3s 86ms/step - loss: 0.1514 - accuracy: 0.9495\n",
            "Epoch 74/75\n",
            "36/36 [==============================] - 3s 87ms/step - loss: 0.1436 - accuracy: 0.9491\n",
            "Epoch 75/75\n",
            "36/36 [==============================] - 3s 86ms/step - loss: 0.0465 - accuracy: 0.9814\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7e2a2d01dcc0>"
            ]
          },
          "metadata": {},
          "execution_count": 164
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cnn.predict(xtest)\n",
        "pred = (pred > 0.5)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "tTCpubuPftCc",
        "outputId": "f9a44771-71f0-4884-feab-5684313d154f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "31/31 [==============================] - 1s 16ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(ytest, pred), precision_score(ytest, pred), recall_score(ytest, pred), f1_score(ytest, pred), cohen_kappa_score(ytest, pred), matthews_corrcoef(ytest, pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "RdQTkyr5fx9B",
        "outputId": "6cfd79d0-0881-4d61-9eaf-0753deba0be6"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9742002063983488,\n",
              " 0.966824644549763,\n",
              " 0.918918918918919,\n",
              " 0.9422632794457275,\n",
              " 0.9256657860095184,\n",
              " 0.9261612997336975)"
            ]
          },
          "metadata": {},
          "execution_count": 166
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(ytest, pred)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 210
        },
        "id": "XABcSJppf_zE",
        "outputId": "f16876e6-a2b4-4d44-eb9e-25f2d54a3eda"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "error",
          "ename": "NameError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-19-b4f20019a871>\u001b[0m in \u001b[0;36m<cell line: 1>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mcm1\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mconfusion_matrix\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mytest\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mpred\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      2\u001b[0m \u001b[0mspecificity\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcm1\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m/\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mcm1\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m+\u001b[0m\u001b[0mcm1\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0mspecificity\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mNameError\u001b[0m: name 'ytest' is not defined"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "HJ26qBd9oidU"
      },
      "source": [
        "**ADASYN**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "q4rpVD9MokNN"
      },
      "outputs": [],
      "source": [
        "df1 = pd.read_csv('/content/LSA_TR.csv')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "m4-q2QyconUk"
      },
      "outputs": [],
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "7eVmFNcfoqNM"
      },
      "outputs": [],
      "source": [
        "from imblearn.over_sampling import ADASYN\n",
        "ada = ADASYN()\n",
        "X, Y = ada.fit_resample(X, Y)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "b9oWi-kkotRl"
      },
      "outputs": [],
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "n4zfd57Loyy7"
      },
      "outputs": [],
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "C81XP8lOo2db"
      },
      "outputs": [],
      "source": [
        "cnn = Sequential()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "lnEB8fqMo5hD"
      },
      "outputs": [],
      "source": [
        "cnn.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))\n",
        "cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))\n",
        "#cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "UzX2-xFWo8xr"
      },
      "outputs": [],
      "source": [
        "cnn.add(MaxPool1D(pool_size=4))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "E5raeE1Do_dz"
      },
      "outputs": [],
      "source": [
        "cnn.add(LSTM(256, activation='relu'))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "eAaQxiS5pCSN"
      },
      "outputs": [],
      "source": [
        "cnn.add(Flatten())"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "hcMqhfkopEwk"
      },
      "outputs": [],
      "source": [
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "l08lB34hpH3j"
      },
      "outputs": [],
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xIQTbk_qpLGu",
        "outputId": "5866e92c-08c7-4128-d7d2-e030004432d4"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "66/66 [==============================] - 12s 145ms/step - loss: 0.6340 - accuracy: 0.6414\n",
            "Epoch 2/75\n",
            "66/66 [==============================] - 9s 139ms/step - loss: 0.5049 - accuracy: 0.7913\n",
            "Epoch 3/75\n",
            "66/66 [==============================] - 8s 126ms/step - loss: 0.4412 - accuracy: 0.8107\n",
            "Epoch 4/75\n",
            "66/66 [==============================] - 9s 142ms/step - loss: 0.2866 - accuracy: 0.8921\n",
            "Epoch 5/75\n",
            "66/66 [==============================] - 9s 143ms/step - loss: 0.2303 - accuracy: 0.9132\n",
            "Epoch 6/75\n",
            "66/66 [==============================] - 9s 132ms/step - loss: 0.1936 - accuracy: 0.9268\n",
            "Epoch 7/75\n",
            "66/66 [==============================] - 9s 128ms/step - loss: 0.1899 - accuracy: 0.9312\n",
            "Epoch 8/75\n",
            "66/66 [==============================] - 6s 96ms/step - loss: 0.1628 - accuracy: 0.9422\n",
            "Epoch 9/75\n",
            "66/66 [==============================] - 6s 91ms/step - loss: 0.1312 - accuracy: 0.9539\n",
            "Epoch 10/75\n",
            "66/66 [==============================] - 6s 90ms/step - loss: 0.1206 - accuracy: 0.9597\n",
            "Epoch 11/75\n",
            "66/66 [==============================] - 7s 99ms/step - loss: 0.0989 - accuracy: 0.9640\n",
            "Epoch 12/75\n",
            "66/66 [==============================] - 6s 86ms/step - loss: 0.1032 - accuracy: 0.9679\n",
            "Epoch 13/75\n",
            "66/66 [==============================] - 7s 104ms/step - loss: 0.1116 - accuracy: 0.9585\n",
            "Epoch 14/75\n",
            "66/66 [==============================] - 6s 91ms/step - loss: 0.0893 - accuracy: 0.9693\n",
            "Epoch 15/75\n",
            "66/66 [==============================] - 7s 104ms/step - loss: 0.0812 - accuracy: 0.9739\n",
            "Epoch 16/75\n",
            "66/66 [==============================] - 6s 85ms/step - loss: 0.0805 - accuracy: 0.9748\n",
            "Epoch 17/75\n",
            "66/66 [==============================] - 7s 103ms/step - loss: 0.0758 - accuracy: 0.9739\n",
            "Epoch 18/75\n",
            "66/66 [==============================] - 6s 85ms/step - loss: 0.0756 - accuracy: 0.9746\n",
            "Epoch 19/75\n",
            "66/66 [==============================] - 7s 103ms/step - loss: 0.0518 - accuracy: 0.9839\n",
            "Epoch 20/75\n",
            "66/66 [==============================] - 6s 85ms/step - loss: 0.0553 - accuracy: 0.9827\n",
            "Epoch 21/75\n",
            "66/66 [==============================] - 7s 104ms/step - loss: 0.0440 - accuracy: 0.9878\n",
            "Epoch 22/75\n",
            "66/66 [==============================] - 7s 103ms/step - loss: 0.0257 - accuracy: 0.9921\n",
            "Epoch 23/75\n",
            "66/66 [==============================] - 7s 104ms/step - loss: 0.0368 - accuracy: 0.9892\n",
            "Epoch 24/75\n",
            "66/66 [==============================] - 6s 85ms/step - loss: 0.0331 - accuracy: 0.9909\n",
            "Epoch 25/75\n",
            "66/66 [==============================] - 7s 104ms/step - loss: 0.0408 - accuracy: 0.9861\n",
            "Epoch 26/75\n",
            "66/66 [==============================] - 6s 86ms/step - loss: 0.0538 - accuracy: 0.9844\n",
            "Epoch 27/75\n",
            "66/66 [==============================] - 7s 103ms/step - loss: 0.0288 - accuracy: 0.9909\n",
            "Epoch 28/75\n",
            "66/66 [==============================] - 6s 85ms/step - loss: 0.0492 - accuracy: 0.9822\n",
            "Epoch 29/75\n",
            "66/66 [==============================] - 7s 105ms/step - loss: 0.0212 - accuracy: 0.9935\n",
            "Epoch 30/75\n",
            "66/66 [==============================] - 6s 86ms/step - loss: 0.0706 - accuracy: 0.9763\n",
            "Epoch 31/75\n",
            "66/66 [==============================] - 7s 111ms/step - loss: 0.1390 - accuracy: 0.9611\n",
            "Epoch 32/75\n",
            "66/66 [==============================] - 6s 85ms/step - loss: 0.0385 - accuracy: 0.9870\n",
            "Epoch 33/75\n",
            "66/66 [==============================] - 7s 103ms/step - loss: 0.0199 - accuracy: 0.9933\n",
            "Epoch 34/75\n",
            "66/66 [==============================] - 6s 86ms/step - loss: 0.0189 - accuracy: 0.9942\n",
            "Epoch 35/75\n",
            "66/66 [==============================] - 7s 100ms/step - loss: 0.0225 - accuracy: 0.9930\n",
            "Epoch 36/75\n",
            "66/66 [==============================] - 6s 88ms/step - loss: 0.0097 - accuracy: 0.9981\n",
            "Epoch 37/75\n",
            "66/66 [==============================] - 6s 93ms/step - loss: 0.0110 - accuracy: 0.9966\n",
            "Epoch 38/75\n",
            "66/66 [==============================] - 6s 94ms/step - loss: 0.0269 - accuracy: 0.9906\n",
            "Epoch 39/75\n",
            "66/66 [==============================] - 6s 87ms/step - loss: 0.0099 - accuracy: 0.9969\n",
            "Epoch 40/75\n",
            "66/66 [==============================] - 7s 100ms/step - loss: 0.0082 - accuracy: 0.9971\n",
            "Epoch 41/75\n",
            "66/66 [==============================] - 6s 84ms/step - loss: 0.0140 - accuracy: 0.9957\n",
            "Epoch 42/75\n",
            "66/66 [==============================] - 7s 104ms/step - loss: 0.0308 - accuracy: 0.9902\n",
            "Epoch 43/75\n",
            "66/66 [==============================] - 6s 84ms/step - loss: 0.0223 - accuracy: 0.9921\n",
            "Epoch 44/75\n",
            "66/66 [==============================] - 7s 103ms/step - loss: 0.0764 - accuracy: 0.9801\n",
            "Epoch 45/75\n",
            "66/66 [==============================] - 6s 85ms/step - loss: 0.0241 - accuracy: 0.9906\n",
            "Epoch 46/75\n",
            "66/66 [==============================] - 7s 104ms/step - loss: 0.0212 - accuracy: 0.9928\n",
            "Epoch 47/75\n",
            "66/66 [==============================] - 6s 87ms/step - loss: 0.0039 - accuracy: 0.9988\n",
            "Epoch 48/75\n",
            "66/66 [==============================] - 8s 125ms/step - loss: 0.0021 - accuracy: 0.9993\n",
            "Epoch 49/75\n",
            "66/66 [==============================] - 6s 85ms/step - loss: 0.0211 - accuracy: 0.9930\n",
            "Epoch 50/75\n",
            "66/66 [==============================] - 7s 103ms/step - loss: 0.0074 - accuracy: 0.9974\n",
            "Epoch 51/75\n",
            "66/66 [==============================] - 6s 86ms/step - loss: 0.0022 - accuracy: 0.9993\n",
            "Epoch 52/75\n",
            "66/66 [==============================] - 7s 100ms/step - loss: 0.0016 - accuracy: 0.9995\n",
            "Epoch 53/75\n",
            "66/66 [==============================] - 6s 88ms/step - loss: 0.0090 - accuracy: 0.9974\n",
            "Epoch 54/75\n",
            "66/66 [==============================] - 6s 96ms/step - loss: 0.0061 - accuracy: 0.9986\n",
            "Epoch 55/75\n",
            "66/66 [==============================] - 6s 92ms/step - loss: 0.0062 - accuracy: 0.9976\n",
            "Epoch 56/75\n",
            "66/66 [==============================] - 6s 90ms/step - loss: 0.0414 - accuracy: 0.9858\n",
            "Epoch 57/75\n",
            "66/66 [==============================] - 6s 98ms/step - loss: 0.0067 - accuracy: 0.9974\n",
            "Epoch 58/75\n",
            "66/66 [==============================] - 6s 85ms/step - loss: 0.0061 - accuracy: 0.9971\n",
            "Epoch 59/75\n",
            "66/66 [==============================] - 7s 104ms/step - loss: 0.0264 - accuracy: 0.9918\n",
            "Epoch 60/75\n",
            "66/66 [==============================] - 6s 86ms/step - loss: 0.0040 - accuracy: 0.9990\n",
            "Epoch 61/75\n",
            "66/66 [==============================] - 7s 104ms/step - loss: 0.0034 - accuracy: 0.9990\n",
            "Epoch 62/75\n",
            "66/66 [==============================] - 6s 85ms/step - loss: 0.0044 - accuracy: 0.9986\n",
            "Epoch 63/75\n",
            "66/66 [==============================] - 7s 103ms/step - loss: 0.0097 - accuracy: 0.9971\n",
            "Epoch 64/75\n",
            "66/66 [==============================] - 6s 86ms/step - loss: 0.0318 - accuracy: 0.9890\n",
            "Epoch 65/75\n",
            "66/66 [==============================] - 7s 104ms/step - loss: 0.0171 - accuracy: 0.9940\n",
            "Epoch 66/75\n",
            "66/66 [==============================] - 6s 85ms/step - loss: 0.0154 - accuracy: 0.9935\n",
            "Epoch 67/75\n",
            "66/66 [==============================] - 7s 104ms/step - loss: 0.0191 - accuracy: 0.9940\n",
            "Epoch 68/75\n",
            "66/66 [==============================] - 6s 85ms/step - loss: 0.0044 - accuracy: 0.9986\n",
            "Epoch 69/75\n",
            "66/66 [==============================] - 7s 104ms/step - loss: 0.0022 - accuracy: 0.9993\n",
            "Epoch 70/75\n",
            "66/66 [==============================] - 6s 85ms/step - loss: 2.6036e-04 - accuracy: 1.0000\n",
            "Epoch 71/75\n",
            "66/66 [==============================] - 7s 104ms/step - loss: 1.3868e-04 - accuracy: 1.0000\n",
            "Epoch 72/75\n",
            "66/66 [==============================] - 6s 85ms/step - loss: 1.0140e-04 - accuracy: 1.0000\n",
            "Epoch 73/75\n",
            "66/66 [==============================] - 7s 105ms/step - loss: 6.9650e-05 - accuracy: 1.0000\n",
            "Epoch 74/75\n",
            "66/66 [==============================] - 6s 94ms/step - loss: 5.3880e-05 - accuracy: 1.0000\n",
            "Epoch 75/75\n",
            "66/66 [==============================] - 8s 113ms/step - loss: 4.4470e-05 - accuracy: 1.0000\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7e2a24111d80>"
            ]
          },
          "metadata": {},
          "execution_count": 181
        }
      ],
      "source": [
        "cnn.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "RWQwm-2spOEk",
        "outputId": "7862b787-5925-47bc-b57e-9507d6b8aaa3"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "33/33 [==============================] - 1s 16ms/step\n"
          ]
        }
      ],
      "source": [
        "pred = cnn.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "DeJ9aoNWpROk",
        "outputId": "a53fe024-fcda-4a48-96e7-db2b24b0c8d8"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9932821497120922,\n",
              " 0.9981916817359855,\n",
              " 0.989247311827957,\n",
              " 0.9865054374169772,\n",
              " 0.9865512686249185)"
            ]
          },
          "metadata": {},
          "execution_count": 183
        }
      ],
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "VJTLWFhlpV4N",
        "outputId": "adfc0cd5-3583-4fe7-b5ce-aa8c683fdfd0"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9877300613496932"
            ]
          },
          "metadata": {},
          "execution_count": 184
        }
      ],
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "CrkeTI46FUU-"
      },
      "source": [
        "**SMOTETomek**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "_A6W2uyyFYO7"
      },
      "outputs": [],
      "source": [
        "df1 = pd.read_csv('/content/LSA_TR.csv')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "wS1229LiFahI"
      },
      "outputs": [],
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "QEZDrtTZFeHv"
      },
      "outputs": [],
      "source": [
        "from imblearn.combine import SMOTETomek\n",
        "smt = SMOTETomek()\n",
        "X, Y = smt.fit_resample(X, Y)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "eGiRg8IOFgX3"
      },
      "outputs": [],
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "L9QQ41wHFim3"
      },
      "outputs": [],
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "sseSaNdXFkPj"
      },
      "outputs": [],
      "source": [
        "cnn = Sequential()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "5rf3seXUFmkv"
      },
      "outputs": [],
      "source": [
        "cnn.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))\n",
        "cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))\n",
        "#cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "lm0rjHXrFoPD"
      },
      "outputs": [],
      "source": [
        "cnn.add(MaxPool1D(pool_size=4))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "NXK249IaFqgT"
      },
      "outputs": [],
      "source": [
        "cnn.add(LSTM(256, activation='relu'))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "SIMXLq3MFswM"
      },
      "outputs": [],
      "source": [
        "cnn.add(Flatten())"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "tnHikbZWFvAV"
      },
      "outputs": [],
      "source": [
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "EBw4FRfyFw0N"
      },
      "outputs": [],
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "o2kBn7d8Fyqj",
        "outputId": "6685ef0a-7daa-4ea9-f71a-72f0bf2d080b"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "65/65 [==============================] - 13s 146ms/step - loss: 0.6366 - accuracy: 0.5990\n",
            "Epoch 2/75\n",
            "65/65 [==============================] - 10s 155ms/step - loss: 0.5269 - accuracy: 0.7481\n",
            "Epoch 3/75\n",
            "65/65 [==============================] - 8s 131ms/step - loss: 0.4201 - accuracy: 0.8028\n",
            "Epoch 4/75\n",
            "65/65 [==============================] - 9s 141ms/step - loss: 0.2474 - accuracy: 0.8974\n",
            "Epoch 5/75\n",
            "65/65 [==============================] - 8s 124ms/step - loss: 0.1833 - accuracy: 0.9293\n",
            "Epoch 6/75\n",
            "65/65 [==============================] - 6s 85ms/step - loss: 0.1623 - accuracy: 0.9397\n",
            "Epoch 7/75\n",
            "65/65 [==============================] - 7s 104ms/step - loss: 0.1354 - accuracy: 0.9504\n",
            "Epoch 8/75\n",
            "65/65 [==============================] - 6s 85ms/step - loss: 0.1025 - accuracy: 0.9620\n",
            "Epoch 9/75\n",
            "65/65 [==============================] - 7s 105ms/step - loss: 0.1052 - accuracy: 0.9647\n",
            "Epoch 10/75\n",
            "65/65 [==============================] - 6s 87ms/step - loss: 0.0733 - accuracy: 0.9719\n",
            "Epoch 11/75\n",
            "65/65 [==============================] - 7s 105ms/step - loss: 0.0595 - accuracy: 0.9806\n",
            "Epoch 12/75\n",
            "65/65 [==============================] - 6s 85ms/step - loss: 0.0448 - accuracy: 0.9879\n",
            "Epoch 13/75\n",
            "65/65 [==============================] - 8s 125ms/step - loss: 0.0724 - accuracy: 0.9741\n",
            "Epoch 14/75\n",
            "65/65 [==============================] - 6s 86ms/step - loss: 0.0411 - accuracy: 0.9848\n",
            "Epoch 15/75\n",
            "65/65 [==============================] - 7s 105ms/step - loss: 0.0340 - accuracy: 0.9891\n",
            "Epoch 16/75\n",
            "65/65 [==============================] - 6s 86ms/step - loss: 0.0385 - accuracy: 0.9877\n",
            "Epoch 17/75\n",
            "65/65 [==============================] - 6s 98ms/step - loss: 0.0320 - accuracy: 0.9884\n",
            "Epoch 18/75\n",
            "65/65 [==============================] - 6s 92ms/step - loss: 0.0208 - accuracy: 0.9944\n",
            "Epoch 19/75\n",
            "65/65 [==============================] - 6s 92ms/step - loss: 0.0447 - accuracy: 0.9862\n",
            "Epoch 20/75\n",
            "65/65 [==============================] - 6s 98ms/step - loss: 0.0251 - accuracy: 0.9898\n",
            "Epoch 21/75\n",
            "65/65 [==============================] - 6s 86ms/step - loss: 0.0311 - accuracy: 0.9884\n",
            "Epoch 22/75\n",
            "65/65 [==============================] - 7s 104ms/step - loss: 0.0166 - accuracy: 0.9937\n",
            "Epoch 23/75\n",
            "65/65 [==============================] - 6s 86ms/step - loss: 0.0129 - accuracy: 0.9956\n",
            "Epoch 24/75\n",
            "65/65 [==============================] - 7s 105ms/step - loss: 0.0180 - accuracy: 0.9942\n",
            "Epoch 25/75\n",
            "65/65 [==============================] - 6s 86ms/step - loss: 0.1245 - accuracy: 0.9586\n",
            "Epoch 26/75\n",
            "65/65 [==============================] - 7s 105ms/step - loss: 0.0292 - accuracy: 0.9894\n",
            "Epoch 27/75\n",
            "65/65 [==============================] - 6s 86ms/step - loss: 0.0123 - accuracy: 0.9964\n",
            "Epoch 28/75\n",
            "65/65 [==============================] - 7s 108ms/step - loss: 0.0118 - accuracy: 0.9952\n",
            "Epoch 29/75\n",
            "65/65 [==============================] - 6s 86ms/step - loss: 0.0156 - accuracy: 0.9959\n",
            "Epoch 30/75\n",
            "65/65 [==============================] - 7s 105ms/step - loss: 0.0105 - accuracy: 0.9966\n",
            "Epoch 31/75\n",
            "65/65 [==============================] - 6s 87ms/step - loss: 0.0330 - accuracy: 0.9874\n",
            "Epoch 32/75\n",
            "65/65 [==============================] - 7s 105ms/step - loss: 0.0104 - accuracy: 0.9971\n",
            "Epoch 33/75\n",
            "65/65 [==============================] - 6s 86ms/step - loss: 0.0057 - accuracy: 0.9990\n",
            "Epoch 34/75\n",
            "65/65 [==============================] - 7s 105ms/step - loss: 0.0326 - accuracy: 0.9889\n",
            "Epoch 35/75\n",
            "65/65 [==============================] - 6s 86ms/step - loss: 0.0024 - accuracy: 0.9998\n",
            "Epoch 36/75\n",
            "65/65 [==============================] - 7s 106ms/step - loss: 0.0049 - accuracy: 0.9983\n",
            "Epoch 37/75\n",
            "65/65 [==============================] - 6s 86ms/step - loss: 0.0174 - accuracy: 0.9939\n",
            "Epoch 38/75\n",
            "65/65 [==============================] - 7s 101ms/step - loss: 0.0025 - accuracy: 0.9995\n",
            "Epoch 39/75\n",
            "65/65 [==============================] - 7s 110ms/step - loss: 5.9741e-04 - accuracy: 0.9998\n",
            "Epoch 40/75\n",
            "65/65 [==============================] - 7s 105ms/step - loss: 0.0190 - accuracy: 0.9927\n",
            "Epoch 41/75\n",
            "65/65 [==============================] - 6s 86ms/step - loss: 0.0122 - accuracy: 0.9959\n",
            "Epoch 42/75\n",
            "65/65 [==============================] - 7s 100ms/step - loss: 0.0152 - accuracy: 0.9959\n",
            "Epoch 43/75\n",
            "65/65 [==============================] - 6s 90ms/step - loss: 0.0038 - accuracy: 0.9988\n",
            "Epoch 44/75\n",
            "65/65 [==============================] - 6s 96ms/step - loss: 0.0017 - accuracy: 0.9998\n",
            "Epoch 45/75\n",
            "65/65 [==============================] - 6s 96ms/step - loss: 0.0125 - accuracy: 0.9964\n",
            "Epoch 46/75\n",
            "65/65 [==============================] - 6s 88ms/step - loss: 0.0013 - accuracy: 0.9998\n",
            "Epoch 47/75\n",
            "65/65 [==============================] - 7s 102ms/step - loss: 0.0323 - accuracy: 0.9903\n",
            "Epoch 48/75\n",
            "65/65 [==============================] - 6s 86ms/step - loss: 0.0060 - accuracy: 0.9985\n",
            "Epoch 49/75\n",
            "65/65 [==============================] - 7s 105ms/step - loss: 0.0013 - accuracy: 0.9995\n",
            "Epoch 50/75\n",
            "65/65 [==============================] - 6s 86ms/step - loss: 5.5251e-04 - accuracy: 0.9998\n",
            "Epoch 51/75\n",
            "65/65 [==============================] - 7s 104ms/step - loss: 3.3105e-04 - accuracy: 1.0000\n",
            "Epoch 52/75\n",
            "65/65 [==============================] - 6s 87ms/step - loss: 6.9295e-05 - accuracy: 1.0000\n",
            "Epoch 53/75\n",
            "65/65 [==============================] - 7s 105ms/step - loss: 4.2279e-05 - accuracy: 1.0000\n",
            "Epoch 54/75\n",
            "65/65 [==============================] - 6s 86ms/step - loss: 3.2414e-05 - accuracy: 1.0000\n",
            "Epoch 55/75\n",
            "65/65 [==============================] - 7s 105ms/step - loss: 2.7047e-05 - accuracy: 1.0000\n",
            "Epoch 56/75\n",
            "65/65 [==============================] - 6s 86ms/step - loss: 2.2915e-05 - accuracy: 1.0000\n",
            "Epoch 57/75\n",
            "65/65 [==============================] - 7s 106ms/step - loss: 1.9857e-05 - accuracy: 1.0000\n",
            "Epoch 58/75\n",
            "65/65 [==============================] - 6s 86ms/step - loss: 1.7534e-05 - accuracy: 1.0000\n",
            "Epoch 59/75\n",
            "65/65 [==============================] - 7s 104ms/step - loss: 1.5414e-05 - accuracy: 1.0000\n",
            "Epoch 60/75\n",
            "65/65 [==============================] - 6s 86ms/step - loss: 1.3813e-05 - accuracy: 1.0000\n",
            "Epoch 61/75\n",
            "65/65 [==============================] - 7s 106ms/step - loss: 1.2438e-05 - accuracy: 1.0000\n",
            "Epoch 62/75\n",
            "65/65 [==============================] - 6s 86ms/step - loss: 1.1353e-05 - accuracy: 1.0000\n",
            "Epoch 63/75\n",
            "65/65 [==============================] - 7s 100ms/step - loss: 1.0276e-05 - accuracy: 1.0000\n",
            "Epoch 64/75\n",
            "65/65 [==============================] - 6s 91ms/step - loss: 9.2792e-06 - accuracy: 1.0000\n",
            "Epoch 65/75\n",
            "65/65 [==============================] - 8s 124ms/step - loss: 8.4346e-06 - accuracy: 1.0000\n",
            "Epoch 66/75\n",
            "65/65 [==============================] - 6s 86ms/step - loss: 7.8164e-06 - accuracy: 1.0000\n",
            "Epoch 67/75\n",
            "65/65 [==============================] - 7s 102ms/step - loss: 7.1620e-06 - accuracy: 1.0000\n",
            "Epoch 68/75\n",
            "65/65 [==============================] - 6s 90ms/step - loss: 6.6086e-06 - accuracy: 1.0000\n",
            "Epoch 69/75\n",
            "65/65 [==============================] - 6s 94ms/step - loss: 6.0760e-06 - accuracy: 1.0000\n",
            "Epoch 70/75\n",
            "65/65 [==============================] - 6s 95ms/step - loss: 5.6163e-06 - accuracy: 1.0000\n",
            "Epoch 71/75\n",
            "65/65 [==============================] - 6s 87ms/step - loss: 5.2042e-06 - accuracy: 1.0000\n",
            "Epoch 72/75\n",
            "65/65 [==============================] - 7s 103ms/step - loss: 4.8722e-06 - accuracy: 1.0000\n",
            "Epoch 73/75\n",
            "65/65 [==============================] - 6s 86ms/step - loss: 4.5103e-06 - accuracy: 1.0000\n",
            "Epoch 74/75\n",
            "65/65 [==============================] - 7s 105ms/step - loss: 4.2187e-06 - accuracy: 1.0000\n",
            "Epoch 75/75\n",
            "65/65 [==============================] - 6s 87ms/step - loss: 3.9306e-06 - accuracy: 1.0000\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7e2a21f41fc0>"
            ]
          },
          "metadata": {},
          "execution_count": 197
        }
      ],
      "source": [
        "cnn.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "pSxQe4j1F0ci",
        "outputId": "f86f8b86-273e-469b-848a-cf49351d0b6b"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "33/33 [==============================] - 1s 18ms/step\n"
          ]
        }
      ],
      "source": [
        "pred = cnn.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "aWO1_gxRF2nD",
        "outputId": "84f86337-5e3a-472b-a072-fa15b65a7fc4"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9874031007751938,\n",
              " 0.9980657640232108,\n",
              " 0.9772727272727273,\n",
              " 0.974805066029657,\n",
              " 0.9750266615438313)"
            ]
          },
          "metadata": {},
          "execution_count": 199
        }
      ],
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "kUK_vc8gF5UD",
        "outputId": "25983f03-8212-4a59-84a6-57931d2ea610"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9766990291262136"
            ]
          },
          "metadata": {},
          "execution_count": 200
        }
      ],
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "LfbBD1wlp9P1"
      },
      "source": [
        "**NearMiss**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "4REEcf9op_9F"
      },
      "outputs": [],
      "source": [
        "df1 = pd.read_csv('/content/LSA_TR.csv')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "eyi-af_TqBM8"
      },
      "outputs": [],
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "i3fmYNGDqEfG"
      },
      "outputs": [],
      "source": [
        "from imblearn.under_sampling import NearMiss\n",
        "nm = NearMiss()\n",
        "X, Y = nm.fit_resample(X, Y)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "7hWaphUqqGq_"
      },
      "outputs": [],
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "WS5f4_p7qIje"
      },
      "outputs": [],
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "qmwPQPB5qL3_"
      },
      "outputs": [],
      "source": [
        "cnn = Sequential()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "H616EOnrqNul"
      },
      "outputs": [],
      "source": [
        "cnn.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))\n",
        "cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))\n",
        "#cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "j_JFkIDUqPjd"
      },
      "outputs": [],
      "source": [
        "cnn.add(MaxPool1D(pool_size=4))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "z_Ddf56NqRul"
      },
      "outputs": [],
      "source": [
        "cnn.add(LSTM(256, activation='relu'))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Fst-Wsg7qT19"
      },
      "outputs": [],
      "source": [
        "cnn.add(Flatten())"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "1H9LbKH1qV_t"
      },
      "outputs": [],
      "source": [
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "-tXxKmUWqXyv"
      },
      "outputs": [],
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7R-rG2BFqZ39",
        "outputId": "097bc34d-23e0-4b60-f3d0-4e7a1310fc52"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "17/17 [==============================] - 6s 125ms/step - loss: 0.6862 - accuracy: 0.5058\n",
            "Epoch 2/75\n",
            "17/17 [==============================] - 3s 185ms/step - loss: 0.5644 - accuracy: 0.6599\n",
            "Epoch 3/75\n",
            "17/17 [==============================] - 2s 125ms/step - loss: 0.5239 - accuracy: 0.8537\n",
            "Epoch 4/75\n",
            "17/17 [==============================] - 2s 128ms/step - loss: 0.4663 - accuracy: 0.8411\n",
            "Epoch 5/75\n",
            "17/17 [==============================] - 2s 122ms/step - loss: 0.4032 - accuracy: 0.8866\n",
            "Epoch 6/75\n",
            "17/17 [==============================] - 2s 122ms/step - loss: 0.3402 - accuracy: 0.8866\n",
            "Epoch 7/75\n",
            "17/17 [==============================] - 2s 133ms/step - loss: 0.3063 - accuracy: 0.8837\n",
            "Epoch 8/75\n",
            "17/17 [==============================] - 3s 173ms/step - loss: 0.3016 - accuracy: 0.8847\n",
            "Epoch 9/75\n",
            "17/17 [==============================] - 2s 128ms/step - loss: 0.2450 - accuracy: 0.9128\n",
            "Epoch 10/75\n",
            "17/17 [==============================] - 2s 147ms/step - loss: 0.2185 - accuracy: 0.9186\n",
            "Epoch 11/75\n",
            "17/17 [==============================] - 3s 165ms/step - loss: 0.1933 - accuracy: 0.9157\n",
            "Epoch 12/75\n",
            "17/17 [==============================] - 3s 148ms/step - loss: 0.2031 - accuracy: 0.9128\n",
            "Epoch 13/75\n",
            "17/17 [==============================] - 3s 183ms/step - loss: 0.1641 - accuracy: 0.9302\n",
            "Epoch 14/75\n",
            "17/17 [==============================] - 2s 130ms/step - loss: 0.1088 - accuracy: 0.9545\n",
            "Epoch 15/75\n",
            "17/17 [==============================] - 2s 129ms/step - loss: 0.0849 - accuracy: 0.9680\n",
            "Epoch 16/75\n",
            "17/17 [==============================] - 2s 126ms/step - loss: 0.1430 - accuracy: 0.9477\n",
            "Epoch 17/75\n",
            "17/17 [==============================] - 2s 127ms/step - loss: 0.1293 - accuracy: 0.9486\n",
            "Epoch 18/75\n",
            "17/17 [==============================] - 3s 148ms/step - loss: 0.0910 - accuracy: 0.9641\n",
            "Epoch 19/75\n",
            "17/17 [==============================] - 3s 155ms/step - loss: 0.0664 - accuracy: 0.9700\n",
            "Epoch 20/75\n",
            "17/17 [==============================] - 2s 133ms/step - loss: 0.0485 - accuracy: 0.9806\n",
            "Epoch 21/75\n",
            "17/17 [==============================] - 2s 129ms/step - loss: 0.0626 - accuracy: 0.9748\n",
            "Epoch 22/75\n",
            "17/17 [==============================] - 2s 128ms/step - loss: 0.0603 - accuracy: 0.9719\n",
            "Epoch 23/75\n",
            "17/17 [==============================] - 2s 126ms/step - loss: 0.0624 - accuracy: 0.9738\n",
            "Epoch 24/75\n",
            "17/17 [==============================] - 3s 177ms/step - loss: 0.0487 - accuracy: 0.9816\n",
            "Epoch 25/75\n",
            "17/17 [==============================] - 2s 112ms/step - loss: 0.0447 - accuracy: 0.9816\n",
            "Epoch 26/75\n",
            "17/17 [==============================] - 1s 87ms/step - loss: 0.0843 - accuracy: 0.9583\n",
            "Epoch 27/75\n",
            "17/17 [==============================] - 1s 82ms/step - loss: 0.0698 - accuracy: 0.9719\n",
            "Epoch 28/75\n",
            "17/17 [==============================] - 1s 82ms/step - loss: 0.0486 - accuracy: 0.9806\n",
            "Epoch 29/75\n",
            "17/17 [==============================] - 1s 82ms/step - loss: 0.0221 - accuracy: 0.9922\n",
            "Epoch 30/75\n",
            "17/17 [==============================] - 1s 81ms/step - loss: 0.0298 - accuracy: 0.9913\n",
            "Epoch 31/75\n",
            "17/17 [==============================] - 1s 83ms/step - loss: 0.0291 - accuracy: 0.9864\n",
            "Epoch 32/75\n",
            "17/17 [==============================] - 2s 137ms/step - loss: 0.0195 - accuracy: 0.9932\n",
            "Epoch 33/75\n",
            "17/17 [==============================] - 3s 147ms/step - loss: 0.0373 - accuracy: 0.9845\n",
            "Epoch 34/75\n",
            "17/17 [==============================] - 2s 116ms/step - loss: 0.0524 - accuracy: 0.9806\n",
            "Epoch 35/75\n",
            "17/17 [==============================] - 2s 92ms/step - loss: 0.0306 - accuracy: 0.9913\n",
            "Epoch 36/75\n",
            "17/17 [==============================] - 1s 83ms/step - loss: 0.0107 - accuracy: 0.9961\n",
            "Epoch 37/75\n",
            "17/17 [==============================] - 1s 83ms/step - loss: 0.0119 - accuracy: 0.9942\n",
            "Epoch 38/75\n",
            "17/17 [==============================] - 1s 83ms/step - loss: 0.0193 - accuracy: 0.9922\n",
            "Epoch 39/75\n",
            "17/17 [==============================] - 1s 84ms/step - loss: 0.0344 - accuracy: 0.9855\n",
            "Epoch 40/75\n",
            "17/17 [==============================] - 2s 134ms/step - loss: 0.0136 - accuracy: 0.9952\n",
            "Epoch 41/75\n",
            "17/17 [==============================] - 2s 104ms/step - loss: 0.0067 - accuracy: 0.9961\n",
            "Epoch 42/75\n",
            "17/17 [==============================] - 1s 82ms/step - loss: 0.0035 - accuracy: 0.9981\n",
            "Epoch 43/75\n",
            "17/17 [==============================] - 1s 83ms/step - loss: 0.0016 - accuracy: 1.0000\n",
            "Epoch 44/75\n",
            "17/17 [==============================] - 1s 83ms/step - loss: 0.0019 - accuracy: 0.9990\n",
            "Epoch 45/75\n",
            "17/17 [==============================] - 1s 84ms/step - loss: 0.0019 - accuracy: 1.0000\n",
            "Epoch 46/75\n",
            "17/17 [==============================] - 1s 83ms/step - loss: 0.0074 - accuracy: 0.9971\n",
            "Epoch 47/75\n",
            "17/17 [==============================] - 1s 82ms/step - loss: 0.0079 - accuracy: 0.9952\n",
            "Epoch 48/75\n",
            "17/17 [==============================] - 2s 115ms/step - loss: 0.0015 - accuracy: 1.0000\n",
            "Epoch 49/75\n",
            "17/17 [==============================] - 2s 124ms/step - loss: 5.2959e-04 - accuracy: 1.0000\n",
            "Epoch 50/75\n",
            "17/17 [==============================] - 1s 84ms/step - loss: 2.0839e-04 - accuracy: 1.0000\n",
            "Epoch 51/75\n",
            "17/17 [==============================] - 1s 82ms/step - loss: 2.1170e-04 - accuracy: 1.0000\n",
            "Epoch 52/75\n",
            "17/17 [==============================] - 1s 82ms/step - loss: 1.1381e-04 - accuracy: 1.0000\n",
            "Epoch 53/75\n",
            "17/17 [==============================] - 1s 81ms/step - loss: 8.0881e-05 - accuracy: 1.0000\n",
            "Epoch 54/75\n",
            "17/17 [==============================] - 1s 83ms/step - loss: 7.3008e-05 - accuracy: 1.0000\n",
            "Epoch 55/75\n",
            "17/17 [==============================] - 1s 82ms/step - loss: 5.9151e-05 - accuracy: 1.0000\n",
            "Epoch 56/75\n",
            "17/17 [==============================] - 2s 93ms/step - loss: 5.2532e-05 - accuracy: 1.0000\n",
            "Epoch 57/75\n",
            "17/17 [==============================] - 2s 139ms/step - loss: 4.6262e-05 - accuracy: 1.0000\n",
            "Epoch 58/75\n",
            "17/17 [==============================] - 2s 85ms/step - loss: 4.1070e-05 - accuracy: 1.0000\n",
            "Epoch 59/75\n",
            "17/17 [==============================] - 1s 83ms/step - loss: 3.8029e-05 - accuracy: 1.0000\n",
            "Epoch 60/75\n",
            "17/17 [==============================] - 1s 81ms/step - loss: 3.3431e-05 - accuracy: 1.0000\n",
            "Epoch 61/75\n",
            "17/17 [==============================] - 1s 84ms/step - loss: 3.0362e-05 - accuracy: 1.0000\n",
            "Epoch 62/75\n",
            "17/17 [==============================] - 1s 84ms/step - loss: 2.7542e-05 - accuracy: 1.0000\n",
            "Epoch 63/75\n",
            "17/17 [==============================] - 1s 83ms/step - loss: 2.5789e-05 - accuracy: 1.0000\n",
            "Epoch 64/75\n",
            "17/17 [==============================] - 1s 85ms/step - loss: 2.3596e-05 - accuracy: 1.0000\n",
            "Epoch 65/75\n",
            "17/17 [==============================] - 2s 134ms/step - loss: 2.2248e-05 - accuracy: 1.0000\n",
            "Epoch 66/75\n",
            "17/17 [==============================] - 2s 103ms/step - loss: 2.0094e-05 - accuracy: 1.0000\n",
            "Epoch 67/75\n",
            "17/17 [==============================] - 1s 83ms/step - loss: 1.8477e-05 - accuracy: 1.0000\n",
            "Epoch 68/75\n",
            "17/17 [==============================] - 1s 84ms/step - loss: 1.7178e-05 - accuracy: 1.0000\n",
            "Epoch 69/75\n",
            "17/17 [==============================] - 1s 83ms/step - loss: 1.6112e-05 - accuracy: 1.0000\n",
            "Epoch 70/75\n",
            "17/17 [==============================] - 1s 84ms/step - loss: 1.5009e-05 - accuracy: 1.0000\n",
            "Epoch 71/75\n",
            "17/17 [==============================] - 1s 85ms/step - loss: 1.3859e-05 - accuracy: 1.0000\n",
            "Epoch 72/75\n",
            "17/17 [==============================] - 1s 84ms/step - loss: 1.3036e-05 - accuracy: 1.0000\n",
            "Epoch 73/75\n",
            "17/17 [==============================] - 2s 124ms/step - loss: 1.2620e-05 - accuracy: 1.0000\n",
            "Epoch 74/75\n",
            "17/17 [==============================] - 2s 116ms/step - loss: 1.1821e-05 - accuracy: 1.0000\n",
            "Epoch 75/75\n",
            "17/17 [==============================] - 1s 83ms/step - loss: 1.0030e-05 - accuracy: 1.0000\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7e2a2034feb0>"
            ]
          },
          "metadata": {},
          "execution_count": 213
        }
      ],
      "source": [
        "cnn.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vUly-TN1qcsE",
        "outputId": "f0c57795-c598-4c45-c954-4fc1cd929e11"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "9/9 [==============================] - 0s 16ms/step\n"
          ]
        }
      ],
      "source": [
        "pred = cnn.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "rwfa8YIOqfWN",
        "outputId": "65b1718a-a8ee-46c9-ee65-1a3345c42f60"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(1.0, 1.0, 1.0, 1.0, 1.0)"
            ]
          },
          "metadata": {},
          "execution_count": 215
        }
      ],
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bWAHXuNSqiLo",
        "outputId": "8b336757-07a6-4787-f7db-e8ac126b3d1d"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "1.0"
            ]
          },
          "metadata": {},
          "execution_count": 216
        }
      ],
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "8CkwkgKyMvRS"
      },
      "source": [
        "# **CNN+LSTM(NMBroto)**"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Imbalanced**"
      ],
      "metadata": {
        "id": "oleXqsSWAo9-"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "NLfMG0csMx_J"
      },
      "outputs": [],
      "source": [
        "df2 = pd.read_csv('/content/NMB-TR.csv')\n",
        "columns = df2.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df2[columns]\n",
        "Y = df2[target]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "VqIu0_f5M_UR"
      },
      "outputs": [],
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "sl0LazVbNDwx"
      },
      "outputs": [],
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "o0Ezg7GVNNv-"
      },
      "outputs": [],
      "source": [
        "cnn = Sequential()\n",
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))\n",
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu'))\n",
        "#cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "yWSqCcD2NVRS"
      },
      "outputs": [],
      "source": [
        "cnn.add(MaxPool1D(pool_size=2))\n",
        "cnn.add(LSTM(128, activation='relu'))\n",
        "cnn.add(Flatten())\n",
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "-k98tzlANhoj"
      },
      "outputs": [],
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "jo5lomjqNlZj",
        "outputId": "eb95c262-a435-4f47-9ca0-ac9f9d067760"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Epoch 1/75\n",
            "41/41 [==============================] - 5s 65ms/step - loss: 0.5529 - accuracy: 0.7874\n",
            "Epoch 2/75\n",
            "41/41 [==============================] - 3s 64ms/step - loss: 0.5096 - accuracy: 0.7982\n",
            "Epoch 3/75\n",
            "41/41 [==============================] - 3s 80ms/step - loss: 0.5042 - accuracy: 0.7982\n",
            "Epoch 4/75\n",
            "41/41 [==============================] - 3s 75ms/step - loss: 0.4972 - accuracy: 0.7982\n",
            "Epoch 5/75\n",
            "41/41 [==============================] - 3s 64ms/step - loss: 0.4927 - accuracy: 0.7982\n",
            "Epoch 6/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.4863 - accuracy: 0.7982\n",
            "Epoch 7/75\n",
            "41/41 [==============================] - 3s 64ms/step - loss: 0.4781 - accuracy: 0.7982\n",
            "Epoch 8/75\n",
            "41/41 [==============================] - 4s 92ms/step - loss: 0.4686 - accuracy: 0.7982\n",
            "Epoch 9/75\n",
            "41/41 [==============================] - 3s 63ms/step - loss: 0.4609 - accuracy: 0.7982\n",
            "Epoch 10/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.4558 - accuracy: 0.8005\n",
            "Epoch 11/75\n",
            "41/41 [==============================] - 3s 64ms/step - loss: 0.4438 - accuracy: 0.8002\n",
            "Epoch 12/75\n",
            "41/41 [==============================] - 3s 84ms/step - loss: 0.4407 - accuracy: 0.8110\n",
            "Epoch 13/75\n",
            "41/41 [==============================] - 3s 71ms/step - loss: 0.4178 - accuracy: 0.8187\n",
            "Epoch 14/75\n",
            "41/41 [==============================] - 3s 64ms/step - loss: 0.3908 - accuracy: 0.8296\n",
            "Epoch 15/75\n",
            "41/41 [==============================] - 3s 64ms/step - loss: 0.3722 - accuracy: 0.8400\n",
            "Epoch 16/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.3667 - accuracy: 0.8482\n",
            "Epoch 17/75\n",
            "41/41 [==============================] - 4s 91ms/step - loss: 0.3465 - accuracy: 0.8563\n",
            "Epoch 18/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.3491 - accuracy: 0.8536\n",
            "Epoch 19/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.3162 - accuracy: 0.8648\n",
            "Epoch 20/75\n",
            "41/41 [==============================] - 3s 64ms/step - loss: 0.3000 - accuracy: 0.8819\n",
            "Epoch 21/75\n",
            "41/41 [==============================] - 4s 90ms/step - loss: 0.2769 - accuracy: 0.8904\n",
            "Epoch 22/75\n",
            "41/41 [==============================] - 3s 66ms/step - loss: 0.2540 - accuracy: 0.9005\n",
            "Epoch 23/75\n",
            "41/41 [==============================] - 3s 64ms/step - loss: 0.2350 - accuracy: 0.9067\n",
            "Epoch 24/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.2205 - accuracy: 0.9183\n",
            "Epoch 25/75\n",
            "41/41 [==============================] - 3s 70ms/step - loss: 0.2164 - accuracy: 0.9160\n",
            "Epoch 26/75\n",
            "41/41 [==============================] - 3s 84ms/step - loss: 0.1899 - accuracy: 0.9299\n",
            "Epoch 27/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.1844 - accuracy: 0.9314\n",
            "Epoch 28/75\n",
            "41/41 [==============================] - 3s 64ms/step - loss: 0.1721 - accuracy: 0.9361\n",
            "Epoch 29/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.1542 - accuracy: 0.9477\n",
            "Epoch 30/75\n",
            "41/41 [==============================] - 4s 92ms/step - loss: 0.1236 - accuracy: 0.9613\n",
            "Epoch 31/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.1095 - accuracy: 0.9628\n",
            "Epoch 32/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.1328 - accuracy: 0.9551\n",
            "Epoch 33/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.0979 - accuracy: 0.9717\n",
            "Epoch 34/75\n",
            "41/41 [==============================] - 3s 80ms/step - loss: 0.1006 - accuracy: 0.9617\n",
            "Epoch 35/75\n",
            "41/41 [==============================] - 3s 77ms/step - loss: 0.0776 - accuracy: 0.9744\n",
            "Epoch 36/75\n",
            "41/41 [==============================] - 3s 77ms/step - loss: 0.0616 - accuracy: 0.9818\n",
            "Epoch 37/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.0798 - accuracy: 0.9764\n",
            "Epoch 38/75\n",
            "41/41 [==============================] - 3s 70ms/step - loss: 0.0704 - accuracy: 0.9779\n",
            "Epoch 39/75\n",
            "41/41 [==============================] - 4s 86ms/step - loss: 0.0632 - accuracy: 0.9779\n",
            "Epoch 40/75\n",
            "41/41 [==============================] - 3s 64ms/step - loss: 0.0516 - accuracy: 0.9791\n",
            "Epoch 41/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.0503 - accuracy: 0.9830\n",
            "Epoch 42/75\n",
            "41/41 [==============================] - 3s 66ms/step - loss: 0.0528 - accuracy: 0.9799\n",
            "Epoch 43/75\n",
            "41/41 [==============================] - 4s 93ms/step - loss: 0.0297 - accuracy: 0.9899\n",
            "Epoch 44/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.0280 - accuracy: 0.9903\n",
            "Epoch 45/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.0389 - accuracy: 0.9880\n",
            "Epoch 46/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.0374 - accuracy: 0.9861\n",
            "Epoch 47/75\n",
            "41/41 [==============================] - 3s 76ms/step - loss: 0.0447 - accuracy: 0.9857\n",
            "Epoch 48/75\n",
            "41/41 [==============================] - 3s 79ms/step - loss: 0.0246 - accuracy: 0.9915\n",
            "Epoch 49/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.0090 - accuracy: 0.9965\n",
            "Epoch 50/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.0051 - accuracy: 0.9985\n",
            "Epoch 51/75\n",
            "41/41 [==============================] - 3s 64ms/step - loss: 0.0037 - accuracy: 0.9988\n",
            "Epoch 52/75\n",
            "41/41 [==============================] - 4s 92ms/step - loss: 0.0059 - accuracy: 0.9977\n",
            "Epoch 53/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.0042 - accuracy: 0.9985\n",
            "Epoch 54/75\n",
            "41/41 [==============================] - 3s 66ms/step - loss: 0.0037 - accuracy: 0.9988\n",
            "Epoch 55/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.0052 - accuracy: 0.9981\n",
            "Epoch 56/75\n",
            "41/41 [==============================] - 3s 86ms/step - loss: 0.0041 - accuracy: 0.9977\n",
            "Epoch 57/75\n",
            "41/41 [==============================] - 3s 71ms/step - loss: 0.0052 - accuracy: 0.9969\n",
            "Epoch 58/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.0031 - accuracy: 0.9988\n",
            "Epoch 59/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.0046 - accuracy: 0.9981\n",
            "Epoch 60/75\n",
            "41/41 [==============================] - 3s 69ms/step - loss: 0.0032 - accuracy: 0.9985\n",
            "Epoch 61/75\n",
            "41/41 [==============================] - 4s 86ms/step - loss: 0.0029 - accuracy: 0.9985\n",
            "Epoch 62/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.0033 - accuracy: 0.9988\n",
            "Epoch 63/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.0049 - accuracy: 0.9981\n",
            "Epoch 64/75\n",
            "41/41 [==============================] - 3s 64ms/step - loss: 0.0051 - accuracy: 0.9981\n",
            "Epoch 65/75\n",
            "41/41 [==============================] - 4s 92ms/step - loss: 0.0033 - accuracy: 0.9988\n",
            "Epoch 66/75\n",
            "41/41 [==============================] - 3s 64ms/step - loss: 0.0029 - accuracy: 0.9988\n",
            "Epoch 67/75\n",
            "41/41 [==============================] - 3s 64ms/step - loss: 0.0028 - accuracy: 0.9985\n",
            "Epoch 68/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.0026 - accuracy: 0.9985\n",
            "Epoch 69/75\n",
            "41/41 [==============================] - 3s 73ms/step - loss: 0.0042 - accuracy: 0.9981\n",
            "Epoch 70/75\n",
            "41/41 [==============================] - 3s 84ms/step - loss: 0.0027 - accuracy: 0.9992\n",
            "Epoch 71/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.0033 - accuracy: 0.9988\n",
            "Epoch 72/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.0156 - accuracy: 0.9942\n",
            "Epoch 73/75\n",
            "41/41 [==============================] - 3s 75ms/step - loss: 0.1048 - accuracy: 0.9659\n",
            "Epoch 74/75\n",
            "41/41 [==============================] - 4s 91ms/step - loss: 0.1038 - accuracy: 0.9636\n",
            "Epoch 75/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.0417 - accuracy: 0.9845\n"
          ]
        },
        {
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x786c6d808640>"
            ]
          },
          "execution_count": 164,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "cnn.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "tfBAR3vKOVNh",
        "outputId": "68dee3ae-8d64-49a2-8121-1b889091f7d3"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "21/21 [==============================] - 0s 9ms/step\n"
          ]
        }
      ],
      "source": [
        "pred = cnn.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "zWycSVXLOZMp",
        "outputId": "f670483c-f03e-414f-cfd7-027021424285"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "(0.9457364341085271,\n",
              " 0.9919354838709677,\n",
              " 0.7834394904458599,\n",
              " 0.8413663225797384,\n",
              " 0.8509382666350289)"
            ]
          },
          "execution_count": 166,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "tM53SBkpOewp",
        "outputId": "7279611d-f3bd-4e96-aa11-3de9eb31343e"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "0.9347408829174664"
            ]
          },
          "execution_count": 167,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Test**"
      ],
      "metadata": {
        "id": "q_1t6EEoAuIh"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df2 = pd.read_csv('/content/NMB-TR.csv')\n",
        "columns = df2.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df2[columns]\n",
        "Y = df2[target]"
      ],
      "metadata": {
        "id": "DR1VnABZAv3y"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.3, random_state = 1)"
      ],
      "metadata": {
        "id": "tRsYNBP5mVAd"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "sample_size = xtrain.shape[0] # number of samples in train set\n",
        "time_steps  = xtrain.shape[1] # number of features in train set\n",
        "input_dimension = 1               # each feature is represented by 1 number"
      ],
      "metadata": {
        "id": "crkPkW4ZA0GR"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_data_reshaped = xtrain.values.reshape(sample_size,time_steps,input_dimension)\n",
        "n_timesteps = train_data_reshaped.shape[1]\n",
        "n_features  = train_data_reshaped.shape[2]"
      ],
      "metadata": {
        "id": "kLX3twEBA2xz"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn = Sequential()"
      ],
      "metadata": {
        "id": "3yPFgLomA4xw"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))\n",
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu'))\n",
        "# cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))"
      ],
      "metadata": {
        "id": "AyQkstHYA60C"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "z0eYAWvfA86Q"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(LSTM(128, activation='relu'))"
      ],
      "metadata": {
        "id": "yR9ZglQZA_WJ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Flatten())"
      ],
      "metadata": {
        "id": "zbFnfNI7BBWi"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "UqvGNAt5BDN8"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "H3vuhuWrBFPZ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.fit(xtrain, ytrain, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "KHfk5P0tBHME",
        "outputId": "8128893f-5ee1-4240-f225-3c5322907aa5"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "36/36 [==============================] - 10s 117ms/step - loss: 0.5375 - accuracy: 0.7958\n",
            "Epoch 2/75\n",
            "36/36 [==============================] - 4s 115ms/step - loss: 0.4882 - accuracy: 0.8127\n",
            "Epoch 3/75\n",
            "36/36 [==============================] - 5s 130ms/step - loss: 0.4827 - accuracy: 0.8127\n",
            "Epoch 4/75\n",
            "36/36 [==============================] - 5s 141ms/step - loss: 0.4837 - accuracy: 0.8127\n",
            "Epoch 5/75\n",
            "36/36 [==============================] - 6s 152ms/step - loss: 0.4766 - accuracy: 0.8127\n",
            "Epoch 6/75\n",
            "36/36 [==============================] - 4s 118ms/step - loss: 0.4740 - accuracy: 0.8127\n",
            "Epoch 7/75\n",
            "36/36 [==============================] - 5s 136ms/step - loss: 0.4640 - accuracy: 0.8127\n",
            "Epoch 8/75\n",
            "36/36 [==============================] - 4s 114ms/step - loss: 0.4603 - accuracy: 0.8127\n",
            "Epoch 9/75\n",
            "36/36 [==============================] - 4s 101ms/step - loss: 0.4519 - accuracy: 0.8127\n",
            "Epoch 10/75\n",
            "36/36 [==============================] - 5s 130ms/step - loss: 0.4443 - accuracy: 0.8140\n",
            "Epoch 11/75\n",
            "36/36 [==============================] - 4s 116ms/step - loss: 0.4346 - accuracy: 0.8206\n",
            "Epoch 12/75\n",
            "36/36 [==============================] - 4s 117ms/step - loss: 0.4292 - accuracy: 0.8220\n",
            "Epoch 13/75\n",
            "36/36 [==============================] - 5s 131ms/step - loss: 0.4131 - accuracy: 0.8335\n",
            "Epoch 14/75\n",
            "36/36 [==============================] - 4s 114ms/step - loss: 0.3931 - accuracy: 0.8415\n",
            "Epoch 15/75\n",
            "36/36 [==============================] - 4s 102ms/step - loss: 0.3866 - accuracy: 0.8459\n",
            "Epoch 16/75\n",
            "36/36 [==============================] - 4s 122ms/step - loss: 0.3771 - accuracy: 0.8530\n",
            "Epoch 17/75\n",
            "36/36 [==============================] - 4s 123ms/step - loss: 0.3543 - accuracy: 0.8561\n",
            "Epoch 18/75\n",
            "36/36 [==============================] - 4s 110ms/step - loss: 0.3644 - accuracy: 0.8596\n",
            "Epoch 19/75\n",
            "36/36 [==============================] - 3s 89ms/step - loss: 0.3434 - accuracy: 0.8596\n",
            "Epoch 20/75\n",
            "36/36 [==============================] - 4s 105ms/step - loss: 0.3324 - accuracy: 0.8663\n",
            "Epoch 21/75\n",
            "36/36 [==============================] - 3s 71ms/step - loss: 0.3365 - accuracy: 0.8649\n",
            "Epoch 22/75\n",
            "36/36 [==============================] - 3s 71ms/step - loss: 0.3187 - accuracy: 0.8729\n",
            "Epoch 23/75\n",
            "36/36 [==============================] - 3s 70ms/step - loss: 0.3020 - accuracy: 0.8866\n",
            "Epoch 24/75\n",
            "36/36 [==============================] - 3s 80ms/step - loss: 0.2869 - accuracy: 0.8862\n",
            "Epoch 25/75\n",
            "36/36 [==============================] - 3s 95ms/step - loss: 0.2861 - accuracy: 0.8760\n",
            "Epoch 26/75\n",
            "36/36 [==============================] - 3s 71ms/step - loss: 0.2667 - accuracy: 0.8897\n",
            "Epoch 27/75\n",
            "36/36 [==============================] - 3s 72ms/step - loss: 0.2703 - accuracy: 0.8924\n",
            "Epoch 28/75\n",
            "36/36 [==============================] - 3s 71ms/step - loss: 0.2421 - accuracy: 0.9066\n",
            "Epoch 29/75\n",
            "36/36 [==============================] - 3s 93ms/step - loss: 0.2166 - accuracy: 0.9132\n",
            "Epoch 30/75\n",
            "36/36 [==============================] - 3s 96ms/step - loss: 0.2199 - accuracy: 0.9194\n",
            "Epoch 31/75\n",
            "36/36 [==============================] - 3s 71ms/step - loss: 0.2006 - accuracy: 0.9238\n",
            "Epoch 32/75\n",
            "36/36 [==============================] - 3s 70ms/step - loss: 0.1843 - accuracy: 0.9353\n",
            "Epoch 33/75\n",
            "36/36 [==============================] - 3s 73ms/step - loss: 0.1812 - accuracy: 0.9340\n",
            "Epoch 34/75\n",
            "36/36 [==============================] - 4s 103ms/step - loss: 0.1606 - accuracy: 0.9398\n",
            "Epoch 35/75\n",
            "36/36 [==============================] - 3s 71ms/step - loss: 0.1407 - accuracy: 0.9486\n",
            "Epoch 36/75\n",
            "36/36 [==============================] - 3s 71ms/step - loss: 0.1511 - accuracy: 0.9491\n",
            "Epoch 37/75\n",
            "36/36 [==============================] - 3s 71ms/step - loss: 0.1322 - accuracy: 0.9513\n",
            "Epoch 38/75\n",
            "36/36 [==============================] - 3s 91ms/step - loss: 0.1076 - accuracy: 0.9668\n",
            "Epoch 39/75\n",
            "36/36 [==============================] - 3s 85ms/step - loss: 0.0870 - accuracy: 0.9699\n",
            "Epoch 40/75\n",
            "36/36 [==============================] - 3s 71ms/step - loss: 0.1196 - accuracy: 0.9588\n",
            "Epoch 41/75\n",
            "36/36 [==============================] - 3s 71ms/step - loss: 0.1481 - accuracy: 0.9438\n",
            "Epoch 42/75\n",
            "36/36 [==============================] - 3s 71ms/step - loss: 0.0953 - accuracy: 0.9681\n",
            "Epoch 43/75\n",
            "36/36 [==============================] - 4s 105ms/step - loss: 0.0740 - accuracy: 0.9779\n",
            "Epoch 44/75\n",
            "36/36 [==============================] - 3s 71ms/step - loss: 0.0635 - accuracy: 0.9801\n",
            "Epoch 45/75\n",
            "36/36 [==============================] - 3s 84ms/step - loss: 0.0596 - accuracy: 0.9823\n",
            "Epoch 46/75\n",
            "36/36 [==============================] - 3s 71ms/step - loss: 0.0530 - accuracy: 0.9827\n",
            "Epoch 47/75\n",
            "36/36 [==============================] - 3s 83ms/step - loss: 0.0429 - accuracy: 0.9863\n",
            "Epoch 48/75\n",
            "36/36 [==============================] - 3s 92ms/step - loss: 0.0428 - accuracy: 0.9858\n",
            "Epoch 49/75\n",
            "36/36 [==============================] - 3s 71ms/step - loss: 0.0474 - accuracy: 0.9863\n",
            "Epoch 50/75\n",
            "36/36 [==============================] - 3s 71ms/step - loss: 0.0663 - accuracy: 0.9774\n",
            "Epoch 51/75\n",
            "36/36 [==============================] - 3s 71ms/step - loss: 0.0520 - accuracy: 0.9805\n",
            "Epoch 52/75\n",
            "36/36 [==============================] - 4s 100ms/step - loss: 0.0314 - accuracy: 0.9920\n",
            "Epoch 53/75\n",
            "36/36 [==============================] - 3s 77ms/step - loss: 0.0452 - accuracy: 0.9858\n",
            "Epoch 54/75\n",
            "36/36 [==============================] - 3s 72ms/step - loss: 0.0318 - accuracy: 0.9920\n",
            "Epoch 55/75\n",
            "36/36 [==============================] - 3s 71ms/step - loss: 0.0234 - accuracy: 0.9951\n",
            "Epoch 56/75\n",
            "36/36 [==============================] - 3s 71ms/step - loss: 0.0506 - accuracy: 0.9876\n",
            "Epoch 57/75\n",
            "36/36 [==============================] - 4s 104ms/step - loss: 0.0658 - accuracy: 0.9774\n",
            "Epoch 58/75\n",
            "36/36 [==============================] - 3s 71ms/step - loss: 0.0486 - accuracy: 0.9903\n",
            "Epoch 59/75\n",
            "36/36 [==============================] - 3s 71ms/step - loss: 0.0497 - accuracy: 0.9854\n",
            "Epoch 60/75\n",
            "36/36 [==============================] - 3s 71ms/step - loss: 0.0398 - accuracy: 0.9889\n",
            "Epoch 61/75\n",
            "36/36 [==============================] - 3s 86ms/step - loss: 0.0250 - accuracy: 0.9907\n",
            "Epoch 62/75\n",
            "36/36 [==============================] - 3s 90ms/step - loss: 0.0205 - accuracy: 0.9947\n",
            "Epoch 63/75\n",
            "36/36 [==============================] - 3s 71ms/step - loss: 0.0131 - accuracy: 0.9969\n",
            "Epoch 64/75\n",
            "36/36 [==============================] - 3s 72ms/step - loss: 0.0125 - accuracy: 0.9978\n",
            "Epoch 65/75\n",
            "36/36 [==============================] - 3s 73ms/step - loss: 0.0198 - accuracy: 0.9947\n",
            "Epoch 66/75\n",
            "36/36 [==============================] - 4s 103ms/step - loss: 0.0220 - accuracy: 0.9907\n",
            "Epoch 67/75\n",
            "36/36 [==============================] - 3s 74ms/step - loss: 0.0174 - accuracy: 0.9938\n",
            "Epoch 68/75\n",
            "36/36 [==============================] - 3s 87ms/step - loss: 0.0208 - accuracy: 0.9934\n",
            "Epoch 69/75\n",
            "36/36 [==============================] - 3s 72ms/step - loss: 0.0114 - accuracy: 0.9965\n",
            "Epoch 70/75\n",
            "36/36 [==============================] - 3s 86ms/step - loss: 0.0236 - accuracy: 0.9938\n",
            "Epoch 71/75\n",
            "36/36 [==============================] - 3s 92ms/step - loss: 0.0124 - accuracy: 0.9965\n",
            "Epoch 72/75\n",
            "36/36 [==============================] - 3s 72ms/step - loss: 0.0082 - accuracy: 0.9973\n",
            "Epoch 73/75\n",
            "36/36 [==============================] - 3s 71ms/step - loss: 0.0064 - accuracy: 0.9982\n",
            "Epoch 74/75\n",
            "36/36 [==============================] - 3s 72ms/step - loss: 0.0110 - accuracy: 0.9978\n",
            "Epoch 75/75\n",
            "36/36 [==============================] - 4s 103ms/step - loss: 0.0055 - accuracy: 0.9982\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7e1f7c4b0d60>"
            ]
          },
          "metadata": {},
          "execution_count": 78
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cnn.predict(xtest)\n",
        "pred = (pred > 0.5)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "sPTuHZarBJXq",
        "outputId": "d9bcfc5b-f23b-4e6b-9374-da1575ed2228"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "31/31 [==============================] - 1s 10ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(ytest, pred), precision_score(ytest, pred), recall_score(ytest, pred), f1_score(ytest, pred), cohen_kappa_score(ytest, pred), matthews_corrcoef(ytest, pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "0GmoHa5JBM3V",
        "outputId": "51f26111-ae6b-4181-8a15-6b896590e1e4"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9277605779153767,\n",
              " 0.7900763358778626,\n",
              " 0.9324324324324325,\n",
              " 0.8553719008264463,\n",
              " 0.8076661335873967,\n",
              " 0.8125887237278647)"
            ]
          },
          "metadata": {},
          "execution_count": 80
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(ytest, pred)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "sIhTiSHpBQPh",
        "outputId": "c76215c2-9bb2-42fa-b806-9aa06bb70a0f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9263721552878179"
            ]
          },
          "metadata": {},
          "execution_count": 81
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "CLSa0I2kCus7"
      },
      "source": [
        "**ADASYN**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "kdF09AojCwwi"
      },
      "outputs": [],
      "source": [
        "df2 = pd.read_csv('/content/NMB-TR.csv')\n",
        "columns = df2.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df2[columns]\n",
        "Y = df2[target]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "dBSIancqC0LL"
      },
      "outputs": [],
      "source": [
        "from imblearn.over_sampling import ADASYN\n",
        "ada = ADASYN(random_state=42)\n",
        "X, Y = ada.fit_resample(X, Y)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "UvGUd0IIC49q"
      },
      "outputs": [],
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "gEzxy6aaC88j"
      },
      "outputs": [],
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Um1RU0ZpDAPj"
      },
      "outputs": [],
      "source": [
        "cnn = Sequential()\n",
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))\n",
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu'))\n",
        "#cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "LRI_lDqjDC_a"
      },
      "outputs": [],
      "source": [
        "cnn.add(MaxPool1D(pool_size=2))\n",
        "cnn.add(LSTM(128, activation='relu'))\n",
        "cnn.add(Flatten())\n",
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "7436OLXpDGCx"
      },
      "outputs": [],
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "iueCRwP9DJD5",
        "outputId": "a578012b-e77f-46ca-9412-5cc5be5abebc"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Epoch 1/75\n",
            "66/66 [==============================] - 9s 104ms/step - loss: 0.6884 - accuracy: 0.5268\n",
            "Epoch 2/75\n",
            "66/66 [==============================] - 7s 98ms/step - loss: 0.6662 - accuracy: 0.6145\n",
            "Epoch 3/75\n",
            "66/66 [==============================] - 7s 111ms/step - loss: 0.6298 - accuracy: 0.6684\n",
            "Epoch 4/75\n",
            "66/66 [==============================] - 5s 78ms/step - loss: 0.5807 - accuracy: 0.7054\n",
            "Epoch 5/75\n",
            "66/66 [==============================] - 6s 85ms/step - loss: 0.5069 - accuracy: 0.7635\n",
            "Epoch 6/75\n",
            "66/66 [==============================] - 5s 81ms/step - loss: 0.4210 - accuracy: 0.8136\n",
            "Epoch 7/75\n",
            "66/66 [==============================] - 4s 64ms/step - loss: 0.3471 - accuracy: 0.8462\n",
            "Epoch 8/75\n",
            "66/66 [==============================] - 5s 81ms/step - loss: 0.2596 - accuracy: 0.8977\n",
            "Epoch 9/75\n",
            "66/66 [==============================] - 4s 65ms/step - loss: 0.2189 - accuracy: 0.9135\n",
            "Epoch 10/75\n",
            "66/66 [==============================] - 4s 67ms/step - loss: 0.1820 - accuracy: 0.9323\n",
            "Epoch 11/75\n",
            "66/66 [==============================] - 5s 79ms/step - loss: 0.1470 - accuracy: 0.9497\n",
            "Epoch 12/75\n",
            "66/66 [==============================] - 4s 65ms/step - loss: 0.0892 - accuracy: 0.9738\n",
            "Epoch 13/75\n",
            "66/66 [==============================] - 6s 91ms/step - loss: 0.1014 - accuracy: 0.9642\n",
            "Epoch 14/75\n",
            "66/66 [==============================] - 4s 65ms/step - loss: 0.0604 - accuracy: 0.9809\n",
            "Epoch 15/75\n",
            "66/66 [==============================] - 4s 65ms/step - loss: 0.0429 - accuracy: 0.9869\n",
            "Epoch 16/75\n",
            "66/66 [==============================] - 5s 82ms/step - loss: 0.0414 - accuracy: 0.9850\n",
            "Epoch 17/75\n",
            "66/66 [==============================] - 4s 65ms/step - loss: 0.0518 - accuracy: 0.9833\n",
            "Epoch 18/75\n",
            "66/66 [==============================] - 4s 64ms/step - loss: 0.0520 - accuracy: 0.9807\n",
            "Epoch 19/75\n",
            "66/66 [==============================] - 5s 82ms/step - loss: 0.0147 - accuracy: 0.9955\n",
            "Epoch 20/75\n",
            "66/66 [==============================] - 4s 64ms/step - loss: 0.0165 - accuracy: 0.9943\n",
            "Epoch 21/75\n",
            "66/66 [==============================] - 5s 71ms/step - loss: 0.0130 - accuracy: 0.9971\n",
            "Epoch 22/75\n",
            "66/66 [==============================] - 5s 75ms/step - loss: 0.0097 - accuracy: 0.9974\n",
            "Epoch 23/75\n",
            "66/66 [==============================] - 4s 65ms/step - loss: 0.0097 - accuracy: 0.9971\n",
            "Epoch 24/75\n",
            "66/66 [==============================] - 5s 82ms/step - loss: 0.0759 - accuracy: 0.9733\n",
            "Epoch 25/75\n",
            "66/66 [==============================] - 5s 71ms/step - loss: 0.0499 - accuracy: 0.9819\n",
            "Epoch 26/75\n",
            "66/66 [==============================] - 4s 65ms/step - loss: 0.0114 - accuracy: 0.9959\n",
            "Epoch 27/75\n",
            "66/66 [==============================] - 5s 82ms/step - loss: 0.0074 - accuracy: 0.9981\n",
            "Epoch 28/75\n",
            "66/66 [==============================] - 4s 65ms/step - loss: 0.0046 - accuracy: 0.9990\n",
            "Epoch 29/75\n",
            "66/66 [==============================] - 4s 67ms/step - loss: 0.0049 - accuracy: 0.9988\n",
            "Epoch 30/75\n",
            "66/66 [==============================] - 5s 80ms/step - loss: 0.0043 - accuracy: 0.9993\n",
            "Epoch 31/75\n",
            "66/66 [==============================] - 4s 65ms/step - loss: 0.0045 - accuracy: 0.9986\n",
            "Epoch 32/75\n",
            "66/66 [==============================] - 5s 78ms/step - loss: 0.0035 - accuracy: 0.9993\n",
            "Epoch 33/75\n",
            "66/66 [==============================] - 4s 68ms/step - loss: 0.0094 - accuracy: 0.9971\n",
            "Epoch 34/75\n",
            "66/66 [==============================] - 4s 65ms/step - loss: 0.0206 - accuracy: 0.9928\n",
            "Epoch 35/75\n",
            "66/66 [==============================] - 5s 81ms/step - loss: 0.0469 - accuracy: 0.9845\n",
            "Epoch 36/75\n",
            "66/66 [==============================] - 5s 71ms/step - loss: 0.0519 - accuracy: 0.9838\n",
            "Epoch 37/75\n",
            "66/66 [==============================] - 4s 64ms/step - loss: 0.0263 - accuracy: 0.9940\n",
            "Epoch 38/75\n",
            "66/66 [==============================] - 5s 82ms/step - loss: 0.0162 - accuracy: 0.9955\n",
            "Epoch 39/75\n",
            "66/66 [==============================] - 4s 65ms/step - loss: 0.0132 - accuracy: 0.9955\n",
            "Epoch 40/75\n",
            "66/66 [==============================] - 5s 72ms/step - loss: 0.0299 - accuracy: 0.9917\n",
            "Epoch 41/75\n",
            "66/66 [==============================] - 5s 75ms/step - loss: 0.0078 - accuracy: 0.9983\n",
            "Epoch 42/75\n",
            "66/66 [==============================] - 4s 65ms/step - loss: 0.0070 - accuracy: 0.9979\n",
            "Epoch 43/75\n",
            "66/66 [==============================] - 5s 82ms/step - loss: 0.0065 - accuracy: 0.9981\n",
            "Epoch 44/75\n",
            "66/66 [==============================] - 4s 66ms/step - loss: 0.0037 - accuracy: 0.9990\n",
            "Epoch 45/75\n",
            "66/66 [==============================] - 4s 64ms/step - loss: 0.0054 - accuracy: 0.9988\n",
            "Epoch 46/75\n",
            "66/66 [==============================] - 5s 82ms/step - loss: 0.0051 - accuracy: 0.9988\n",
            "Epoch 47/75\n",
            "66/66 [==============================] - 4s 65ms/step - loss: 0.0054 - accuracy: 0.9981\n",
            "Epoch 48/75\n",
            "66/66 [==============================] - 5s 73ms/step - loss: 0.0044 - accuracy: 0.9990\n",
            "Epoch 49/75\n",
            "66/66 [==============================] - 5s 79ms/step - loss: 0.0054 - accuracy: 0.9990\n",
            "Epoch 50/75\n",
            "66/66 [==============================] - 4s 65ms/step - loss: 0.0317 - accuracy: 0.9902\n",
            "Epoch 51/75\n",
            "66/66 [==============================] - 5s 78ms/step - loss: 0.0099 - accuracy: 0.9969\n",
            "Epoch 52/75\n",
            "66/66 [==============================] - 5s 68ms/step - loss: 0.0053 - accuracy: 0.9983\n",
            "Epoch 53/75\n",
            "66/66 [==============================] - 4s 65ms/step - loss: 0.0054 - accuracy: 0.9990\n",
            "Epoch 54/75\n",
            "66/66 [==============================] - 5s 82ms/step - loss: 0.0101 - accuracy: 0.9967\n",
            "Epoch 55/75\n",
            "66/66 [==============================] - 4s 65ms/step - loss: 0.0219 - accuracy: 0.9926\n",
            "Epoch 56/75\n",
            "66/66 [==============================] - 4s 65ms/step - loss: 0.0460 - accuracy: 0.9869\n",
            "Epoch 57/75\n",
            "66/66 [==============================] - 5s 82ms/step - loss: 0.0213 - accuracy: 0.9928\n",
            "Epoch 58/75\n",
            "66/66 [==============================] - 5s 71ms/step - loss: 0.0071 - accuracy: 0.9986\n",
            "Epoch 59/75\n",
            "66/66 [==============================] - 5s 75ms/step - loss: 0.0044 - accuracy: 0.9990\n",
            "Epoch 60/75\n",
            "66/66 [==============================] - 5s 73ms/step - loss: 0.0040 - accuracy: 0.9990\n",
            "Epoch 61/75\n",
            "66/66 [==============================] - 4s 64ms/step - loss: 0.0034 - accuracy: 0.9990\n",
            "Epoch 62/75\n",
            "66/66 [==============================] - 6s 87ms/step - loss: 0.0037 - accuracy: 0.9993\n",
            "Epoch 63/75\n",
            "66/66 [==============================] - 4s 65ms/step - loss: 0.0037 - accuracy: 0.9990\n",
            "Epoch 64/75\n",
            "66/66 [==============================] - 4s 66ms/step - loss: 0.0046 - accuracy: 0.9988\n",
            "Epoch 65/75\n",
            "66/66 [==============================] - 5s 82ms/step - loss: 0.0040 - accuracy: 0.9986\n",
            "Epoch 66/75\n",
            "66/66 [==============================] - 4s 65ms/step - loss: 0.0042 - accuracy: 0.9988\n",
            "Epoch 67/75\n",
            "66/66 [==============================] - 4s 68ms/step - loss: 0.0034 - accuracy: 0.9990\n",
            "Epoch 68/75\n",
            "66/66 [==============================] - 5s 78ms/step - loss: 0.0036 - accuracy: 0.9993\n",
            "Epoch 69/75\n",
            "66/66 [==============================] - 4s 64ms/step - loss: 0.0035 - accuracy: 0.9990\n",
            "Epoch 70/75\n",
            "66/66 [==============================] - 5s 79ms/step - loss: 0.0030 - accuracy: 0.9993\n",
            "Epoch 71/75\n",
            "66/66 [==============================] - 5s 68ms/step - loss: 0.0029 - accuracy: 0.9993\n",
            "Epoch 72/75\n",
            "66/66 [==============================] - 4s 65ms/step - loss: 0.0035 - accuracy: 0.9988\n",
            "Epoch 73/75\n",
            "66/66 [==============================] - 5s 81ms/step - loss: 0.0187 - accuracy: 0.9948\n",
            "Epoch 74/75\n",
            "66/66 [==============================] - 4s 65ms/step - loss: 0.0647 - accuracy: 0.9795\n",
            "Epoch 75/75\n",
            "66/66 [==============================] - 4s 64ms/step - loss: 0.0433 - accuracy: 0.9857\n"
          ]
        },
        {
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x786c6462c6a0>"
            ]
          },
          "execution_count": 175,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "cnn.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ih1dUN2SDMd1",
        "outputId": "30d4def3-63d7-4452-b722-ac455e549fc2"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "33/33 [==============================] - 1s 9ms/step\n"
          ]
        }
      ],
      "source": [
        "pred = cnn.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "pt79kYgFDPXZ",
        "outputId": "fddbe450-a082-4c52-febe-ef70fd35a5e4"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "(0.9494274809160306,\n",
              " 0.9961685823754789,\n",
              " 0.9106830122591943,\n",
              " 0.8988895765828084,\n",
              " 0.9028429366644708)"
            ]
          },
          "execution_count": 177,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "JCn7eGvBDSdt",
        "outputId": "5c559043-8208-49dc-d17d-eb454a70ffb6"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "0.903041825095057"
            ]
          },
          "execution_count": 178,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "j057ObXUG53q"
      },
      "source": [
        "**SMOTETomek**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "AjyI5B2VG8Fq"
      },
      "outputs": [],
      "source": [
        "df2 = pd.read_csv('/content/NMB-TR.csv')\n",
        "columns = df2.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df2[columns]\n",
        "Y = df2[target]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "pfTKMiTwG_ma"
      },
      "outputs": [],
      "source": [
        "from imblearn.combine import SMOTETomek\n",
        "smt = SMOTETomek()\n",
        "X, Y = smt.fit_resample(X, Y)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "lPUP6jhJHDRU"
      },
      "outputs": [],
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "jy89VjWnHGZs"
      },
      "outputs": [],
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "exgy65hoHJt8"
      },
      "outputs": [],
      "source": [
        "cnn = Sequential()\n",
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))\n",
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu'))\n",
        "#cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "a1X0RlaTHNTJ"
      },
      "outputs": [],
      "source": [
        "cnn.add(MaxPool1D(pool_size=2))\n",
        "cnn.add(LSTM(128, activation='relu'))\n",
        "cnn.add(Flatten())\n",
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ZjDPoAvdHRPb"
      },
      "outputs": [],
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "iHwGFozFHUPZ",
        "outputId": "6960c335-2d96-411e-eae0-21f1658c87a8"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Epoch 1/75\n",
            "65/65 [==============================] - 9s 83ms/step - loss: 0.6862 - accuracy: 0.5288\n",
            "Epoch 2/75\n",
            "65/65 [==============================] - 5s 81ms/step - loss: 0.6570 - accuracy: 0.6184\n",
            "Epoch 3/75\n",
            "65/65 [==============================] - 6s 87ms/step - loss: 0.5975 - accuracy: 0.6981\n",
            "Epoch 4/75\n",
            "65/65 [==============================] - 5s 72ms/step - loss: 0.5456 - accuracy: 0.7337\n",
            "Epoch 5/75\n",
            "65/65 [==============================] - 7s 107ms/step - loss: 0.4896 - accuracy: 0.7709\n",
            "Epoch 6/75\n",
            "65/65 [==============================] - 6s 94ms/step - loss: 0.4261 - accuracy: 0.8114\n",
            "Epoch 7/75\n",
            "65/65 [==============================] - 5s 79ms/step - loss: 0.3696 - accuracy: 0.8400\n",
            "Epoch 8/75\n",
            "65/65 [==============================] - 5s 81ms/step - loss: 0.3437 - accuracy: 0.8482\n",
            "Epoch 9/75\n",
            "65/65 [==============================] - 4s 63ms/step - loss: 0.2851 - accuracy: 0.8857\n",
            "Epoch 10/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.2107 - accuracy: 0.9148\n",
            "Epoch 11/75\n",
            "65/65 [==============================] - 5s 82ms/step - loss: 0.1668 - accuracy: 0.9341\n",
            "Epoch 12/75\n",
            "65/65 [==============================] - 5s 72ms/step - loss: 0.1498 - accuracy: 0.9436\n",
            "Epoch 13/75\n",
            "65/65 [==============================] - 5s 72ms/step - loss: 0.1009 - accuracy: 0.9632\n",
            "Epoch 14/75\n",
            "65/65 [==============================] - 5s 74ms/step - loss: 0.1052 - accuracy: 0.9591\n",
            "Epoch 15/75\n",
            "65/65 [==============================] - 4s 64ms/step - loss: 0.0655 - accuracy: 0.9785\n",
            "Epoch 16/75\n",
            "65/65 [==============================] - 5s 77ms/step - loss: 0.0400 - accuracy: 0.9881\n",
            "Epoch 17/75\n",
            "65/65 [==============================] - 4s 67ms/step - loss: 0.0556 - accuracy: 0.9797\n",
            "Epoch 18/75\n",
            "65/65 [==============================] - 4s 64ms/step - loss: 0.0304 - accuracy: 0.9898\n",
            "Epoch 19/75\n",
            "65/65 [==============================] - 5s 81ms/step - loss: 0.0252 - accuracy: 0.9923\n",
            "Epoch 20/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.0239 - accuracy: 0.9930\n",
            "Epoch 21/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.0269 - accuracy: 0.9903\n",
            "Epoch 22/75\n",
            "65/65 [==============================] - 5s 81ms/step - loss: 0.0238 - accuracy: 0.9918\n",
            "Epoch 23/75\n",
            "65/65 [==============================] - 4s 64ms/step - loss: 0.0406 - accuracy: 0.9881\n",
            "Epoch 24/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.0248 - accuracy: 0.9913\n",
            "Epoch 25/75\n",
            "65/65 [==============================] - 5s 81ms/step - loss: 0.0342 - accuracy: 0.9915\n",
            "Epoch 26/75\n",
            "65/65 [==============================] - 4s 64ms/step - loss: 0.0095 - accuracy: 0.9983\n",
            "Epoch 27/75\n",
            "65/65 [==============================] - 4s 68ms/step - loss: 0.0108 - accuracy: 0.9973\n",
            "Epoch 28/75\n",
            "65/65 [==============================] - 5s 76ms/step - loss: 0.0063 - accuracy: 0.9985\n",
            "Epoch 29/75\n",
            "65/65 [==============================] - 4s 64ms/step - loss: 0.0038 - accuracy: 0.9990\n",
            "Epoch 30/75\n",
            "65/65 [==============================] - 5s 79ms/step - loss: 0.0048 - accuracy: 0.9988\n",
            "Epoch 31/75\n",
            "65/65 [==============================] - 4s 66ms/step - loss: 0.0054 - accuracy: 0.9983\n",
            "Epoch 32/75\n",
            "65/65 [==============================] - 4s 64ms/step - loss: 0.0053 - accuracy: 0.9988\n",
            "Epoch 33/75\n",
            "65/65 [==============================] - 5s 81ms/step - loss: 0.0036 - accuracy: 0.9993\n",
            "Epoch 34/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.0055 - accuracy: 0.9988\n",
            "Epoch 35/75\n",
            "65/65 [==============================] - 5s 72ms/step - loss: 0.0287 - accuracy: 0.9908\n",
            "Epoch 36/75\n",
            "65/65 [==============================] - 5s 81ms/step - loss: 0.0634 - accuracy: 0.9811\n",
            "Epoch 37/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.0165 - accuracy: 0.9952\n",
            "Epoch 38/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.0088 - accuracy: 0.9981\n",
            "Epoch 39/75\n",
            "65/65 [==============================] - 5s 81ms/step - loss: 0.0054 - accuracy: 0.9985\n",
            "Epoch 40/75\n",
            "65/65 [==============================] - 4s 64ms/step - loss: 0.0074 - accuracy: 0.9978\n",
            "Epoch 41/75\n",
            "65/65 [==============================] - 5s 74ms/step - loss: 0.0046 - accuracy: 0.9985\n",
            "Epoch 42/75\n",
            "65/65 [==============================] - 5s 72ms/step - loss: 0.0046 - accuracy: 0.9995\n",
            "Epoch 43/75\n",
            "65/65 [==============================] - 4s 64ms/step - loss: 0.0027 - accuracy: 0.9995\n",
            "Epoch 44/75\n",
            "65/65 [==============================] - 5s 81ms/step - loss: 0.0030 - accuracy: 0.9995\n",
            "Epoch 45/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.0029 - accuracy: 0.9995\n",
            "Epoch 46/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.0031 - accuracy: 0.9993\n",
            "Epoch 47/75\n",
            "65/65 [==============================] - 5s 81ms/step - loss: 0.0033 - accuracy: 0.9993\n",
            "Epoch 48/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.0027 - accuracy: 0.9995\n",
            "Epoch 49/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.0024 - accuracy: 0.9995\n",
            "Epoch 50/75\n",
            "65/65 [==============================] - 5s 80ms/step - loss: 0.0026 - accuracy: 0.9995\n",
            "Epoch 51/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.0026 - accuracy: 0.9995\n",
            "Epoch 52/75\n",
            "65/65 [==============================] - 4s 67ms/step - loss: 0.0031 - accuracy: 0.9990\n",
            "Epoch 53/75\n",
            "65/65 [==============================] - 5s 79ms/step - loss: 0.0030 - accuracy: 0.9993\n",
            "Epoch 54/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.0030 - accuracy: 0.9995\n",
            "Epoch 55/75\n",
            "65/65 [==============================] - 5s 74ms/step - loss: 0.0026 - accuracy: 0.9995\n",
            "Epoch 56/75\n",
            "65/65 [==============================] - 5s 70ms/step - loss: 0.0024 - accuracy: 0.9995\n",
            "Epoch 57/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.0027 - accuracy: 0.9995\n",
            "Epoch 58/75\n",
            "65/65 [==============================] - 6s 91ms/step - loss: 0.0027 - accuracy: 0.9995\n",
            "Epoch 59/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.0025 - accuracy: 0.9995\n",
            "Epoch 60/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.0025 - accuracy: 0.9995\n",
            "Epoch 61/75\n",
            "65/65 [==============================] - 5s 81ms/step - loss: 0.0027 - accuracy: 0.9995\n",
            "Epoch 62/75\n",
            "65/65 [==============================] - 4s 64ms/step - loss: 0.0035 - accuracy: 0.9990\n",
            "Epoch 63/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.0042 - accuracy: 0.9990\n",
            "Epoch 64/75\n",
            "65/65 [==============================] - 5s 82ms/step - loss: 0.0427 - accuracy: 0.9869\n",
            "Epoch 65/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.0950 - accuracy: 0.9676\n",
            "Epoch 66/75\n",
            "65/65 [==============================] - 5s 71ms/step - loss: 0.0256 - accuracy: 0.9915\n",
            "Epoch 67/75\n",
            "65/65 [==============================] - 5s 74ms/step - loss: 0.0146 - accuracy: 0.9947\n",
            "Epoch 68/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.0195 - accuracy: 0.9923\n",
            "Epoch 69/75\n",
            "65/65 [==============================] - 5s 80ms/step - loss: 0.0086 - accuracy: 0.9976\n",
            "Epoch 70/75\n",
            "65/65 [==============================] - 4s 66ms/step - loss: 0.0087 - accuracy: 0.9983\n",
            "Epoch 71/75\n",
            "65/65 [==============================] - 4s 64ms/step - loss: 0.0117 - accuracy: 0.9964\n",
            "Epoch 72/75\n",
            "65/65 [==============================] - 5s 82ms/step - loss: 0.0039 - accuracy: 0.9990\n",
            "Epoch 73/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.0031 - accuracy: 0.9995\n",
            "Epoch 74/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.0027 - accuracy: 0.9995\n",
            "Epoch 75/75\n",
            "65/65 [==============================] - 5s 82ms/step - loss: 0.0025 - accuracy: 0.9995\n"
          ]
        },
        {
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x786c6d5a7490>"
            ]
          },
          "execution_count": 186,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "cnn.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "935mGksJHXTc",
        "outputId": "4f4e6ae2-88e5-48a3-c67b-7f3ec312b765"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "33/33 [==============================] - 0s 8ms/step\n"
          ]
        }
      ],
      "source": [
        "pred = cnn.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "XSwbgTHOHaAr",
        "outputId": "9fa5e106-e6ff-4568-f52a-33fff0a65e0a"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "(0.9815891472868217,\n",
              " 1.0,\n",
              " 0.9646182495344506,\n",
              " 0.9631724852937862,\n",
              " 0.9638263094081329)"
            ]
          },
          "execution_count": 188,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "TyMfh_X4Hd57",
        "outputId": "ee775215-f308-4579-b53e-820ad3d98a18"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "0.9630350194552529"
            ]
          },
          "execution_count": 189,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "EOt0cKvksEeA"
      },
      "source": [
        "**NearMiss**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "rzSNivyisGwW"
      },
      "outputs": [],
      "source": [
        "df2 = pd.read_csv('/content/NMB-TR.csv')\n",
        "columns = df2.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df2[columns]\n",
        "Y = df2[target]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "3To4VBeDsJ4V"
      },
      "outputs": [],
      "source": [
        "from imblearn.under_sampling import NearMiss\n",
        "nm = NearMiss()\n",
        "X, Y = nm.fit_resample(X, Y)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ysfMg9vVsMrf"
      },
      "outputs": [],
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "VrbD5G8KsRLA"
      },
      "outputs": [],
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "zYKOq7MSsUQ2"
      },
      "outputs": [],
      "source": [
        "cnn = Sequential()\n",
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))\n",
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu'))\n",
        "#cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "9Ltb0XjMsXL9"
      },
      "outputs": [],
      "source": [
        "cnn.add(MaxPool1D(pool_size=2))\n",
        "cnn.add(LSTM(128, activation='relu'))\n",
        "cnn.add(Flatten())\n",
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "DohTcJm0saAN"
      },
      "outputs": [],
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "2PdrxObYsc7t",
        "outputId": "6da3072e-478a-4522-a763-499ad3dbf1e6"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Epoch 1/70\n",
            "17/17 [==============================] - 5s 67ms/step - loss: 0.6703 - accuracy: 0.5010\n",
            "Epoch 2/70\n",
            "17/17 [==============================] - 1s 66ms/step - loss: 0.5871 - accuracy: 0.6269\n",
            "Epoch 3/70\n",
            "17/17 [==============================] - 1s 65ms/step - loss: 0.5822 - accuracy: 0.7578\n",
            "Epoch 4/70\n",
            "17/17 [==============================] - 1s 65ms/step - loss: 0.5415 - accuracy: 0.7403\n",
            "Epoch 5/70\n",
            "17/17 [==============================] - 1s 66ms/step - loss: 0.5051 - accuracy: 0.7422\n",
            "Epoch 6/70\n",
            "17/17 [==============================] - 1s 65ms/step - loss: 0.4780 - accuracy: 0.7607\n",
            "Epoch 7/70\n",
            "17/17 [==============================] - 1s 66ms/step - loss: 0.4708 - accuracy: 0.7636\n",
            "Epoch 8/70\n",
            "17/17 [==============================] - 1s 66ms/step - loss: 0.4544 - accuracy: 0.7694\n",
            "Epoch 9/70\n",
            "17/17 [==============================] - 1s 65ms/step - loss: 0.4560 - accuracy: 0.7607\n",
            "Epoch 10/70\n",
            "17/17 [==============================] - 2s 107ms/step - loss: 0.4454 - accuracy: 0.7597\n",
            "Epoch 11/70\n",
            "17/17 [==============================] - 2s 96ms/step - loss: 0.4421 - accuracy: 0.7849\n",
            "Epoch 12/70\n",
            "17/17 [==============================] - 1s 65ms/step - loss: 0.4352 - accuracy: 0.7684\n",
            "Epoch 13/70\n",
            "17/17 [==============================] - 1s 65ms/step - loss: 0.4298 - accuracy: 0.7723\n",
            "Epoch 14/70\n",
            "17/17 [==============================] - 1s 66ms/step - loss: 0.4197 - accuracy: 0.7665\n",
            "Epoch 15/70\n",
            "17/17 [==============================] - 1s 66ms/step - loss: 0.4066 - accuracy: 0.7781\n",
            "Epoch 16/70\n",
            "17/17 [==============================] - 1s 66ms/step - loss: 0.3899 - accuracy: 0.8023\n",
            "Epoch 17/70\n",
            "17/17 [==============================] - 1s 65ms/step - loss: 0.3809 - accuracy: 0.8081\n",
            "Epoch 18/70\n",
            "17/17 [==============================] - 1s 65ms/step - loss: 0.3648 - accuracy: 0.8091\n",
            "Epoch 19/70\n",
            "17/17 [==============================] - 1s 65ms/step - loss: 0.3540 - accuracy: 0.8188\n",
            "Epoch 20/70\n",
            "17/17 [==============================] - 1s 79ms/step - loss: 0.3515 - accuracy: 0.8285\n",
            "Epoch 21/70\n",
            "17/17 [==============================] - 2s 107ms/step - loss: 0.3259 - accuracy: 0.8440\n",
            "Epoch 22/70\n",
            "17/17 [==============================] - 1s 77ms/step - loss: 0.3151 - accuracy: 0.8488\n",
            "Epoch 23/70\n",
            "17/17 [==============================] - 1s 65ms/step - loss: 0.2988 - accuracy: 0.8653\n",
            "Epoch 24/70\n",
            "17/17 [==============================] - 1s 66ms/step - loss: 0.2865 - accuracy: 0.8585\n",
            "Epoch 25/70\n",
            "17/17 [==============================] - 1s 66ms/step - loss: 0.2690 - accuracy: 0.8779\n",
            "Epoch 26/70\n",
            "17/17 [==============================] - 1s 66ms/step - loss: 0.2510 - accuracy: 0.8983\n",
            "Epoch 27/70\n",
            "17/17 [==============================] - 1s 65ms/step - loss: 0.2516 - accuracy: 0.8798\n",
            "Epoch 28/70\n",
            "17/17 [==============================] - 1s 66ms/step - loss: 0.2204 - accuracy: 0.9002\n",
            "Epoch 29/70\n",
            "17/17 [==============================] - 1s 66ms/step - loss: 0.2023 - accuracy: 0.9089\n",
            "Epoch 30/70\n",
            "17/17 [==============================] - 1s 66ms/step - loss: 0.1818 - accuracy: 0.9254\n",
            "Epoch 31/70\n",
            "17/17 [==============================] - 2s 102ms/step - loss: 0.2564 - accuracy: 0.8934\n",
            "Epoch 32/70\n",
            "17/17 [==============================] - 2s 103ms/step - loss: 0.2039 - accuracy: 0.9186\n",
            "Epoch 33/70\n",
            "17/17 [==============================] - 1s 66ms/step - loss: 0.1869 - accuracy: 0.9079\n",
            "Epoch 34/70\n",
            "17/17 [==============================] - 1s 66ms/step - loss: 0.1564 - accuracy: 0.9331\n",
            "Epoch 35/70\n",
            "17/17 [==============================] - 1s 66ms/step - loss: 0.1689 - accuracy: 0.9302\n",
            "Epoch 36/70\n",
            "17/17 [==============================] - 1s 66ms/step - loss: 0.1507 - accuracy: 0.9419\n",
            "Epoch 37/70\n",
            "17/17 [==============================] - 1s 65ms/step - loss: 0.1285 - accuracy: 0.9457\n",
            "Epoch 38/70\n",
            "17/17 [==============================] - 1s 66ms/step - loss: 0.1151 - accuracy: 0.9622\n",
            "Epoch 39/70\n",
            "17/17 [==============================] - 1s 67ms/step - loss: 0.1101 - accuracy: 0.9564\n",
            "Epoch 40/70\n",
            "17/17 [==============================] - 1s 66ms/step - loss: 0.0944 - accuracy: 0.9671\n",
            "Epoch 41/70\n",
            "17/17 [==============================] - 1s 76ms/step - loss: 0.2132 - accuracy: 0.9273\n",
            "Epoch 42/70\n",
            "17/17 [==============================] - 2s 107ms/step - loss: 0.3076 - accuracy: 0.8576\n",
            "Epoch 43/70\n",
            "17/17 [==============================] - 1s 86ms/step - loss: 0.2118 - accuracy: 0.9012\n",
            "Epoch 44/70\n",
            "17/17 [==============================] - 1s 65ms/step - loss: 0.1699 - accuracy: 0.9254\n",
            "Epoch 45/70\n",
            "17/17 [==============================] - 1s 65ms/step - loss: 0.1746 - accuracy: 0.9205\n",
            "Epoch 46/70\n",
            "17/17 [==============================] - 1s 65ms/step - loss: 0.1215 - accuracy: 0.9506\n",
            "Epoch 47/70\n",
            "17/17 [==============================] - 1s 65ms/step - loss: 0.0958 - accuracy: 0.9622\n",
            "Epoch 48/70\n",
            "17/17 [==============================] - 1s 63ms/step - loss: 0.0808 - accuracy: 0.9680\n",
            "Epoch 49/70\n",
            "17/17 [==============================] - 1s 65ms/step - loss: 0.1032 - accuracy: 0.9583\n",
            "Epoch 50/70\n",
            "17/17 [==============================] - 1s 64ms/step - loss: 0.0847 - accuracy: 0.9612\n",
            "Epoch 51/70\n",
            "17/17 [==============================] - 1s 66ms/step - loss: 0.0902 - accuracy: 0.9632\n",
            "Epoch 52/70\n",
            "17/17 [==============================] - 1s 89ms/step - loss: 0.0641 - accuracy: 0.9700\n",
            "Epoch 53/70\n",
            "17/17 [==============================] - 2s 105ms/step - loss: 0.0684 - accuracy: 0.9758\n",
            "Epoch 54/70\n",
            "17/17 [==============================] - 1s 71ms/step - loss: 0.1114 - accuracy: 0.9671\n",
            "Epoch 55/70\n",
            "17/17 [==============================] - 1s 69ms/step - loss: 0.1016 - accuracy: 0.9612\n",
            "Epoch 56/70\n",
            "17/17 [==============================] - 1s 66ms/step - loss: 0.1438 - accuracy: 0.9506\n",
            "Epoch 57/70\n",
            "17/17 [==============================] - 1s 66ms/step - loss: 0.1004 - accuracy: 0.9593\n",
            "Epoch 58/70\n",
            "17/17 [==============================] - 1s 66ms/step - loss: 0.0606 - accuracy: 0.9777\n",
            "Epoch 59/70\n",
            "17/17 [==============================] - 1s 66ms/step - loss: 0.0352 - accuracy: 0.9864\n",
            "Epoch 60/70\n",
            "17/17 [==============================] - 1s 66ms/step - loss: 0.0438 - accuracy: 0.9855\n",
            "Epoch 61/70\n",
            "17/17 [==============================] - 1s 66ms/step - loss: 0.0463 - accuracy: 0.9893\n",
            "Epoch 62/70\n",
            "17/17 [==============================] - 1s 66ms/step - loss: 0.0280 - accuracy: 0.9913\n",
            "Epoch 63/70\n",
            "17/17 [==============================] - 2s 105ms/step - loss: 0.0371 - accuracy: 0.9835\n",
            "Epoch 64/70\n",
            "17/17 [==============================] - 2s 94ms/step - loss: 0.0226 - accuracy: 0.9922\n",
            "Epoch 65/70\n",
            "17/17 [==============================] - 1s 66ms/step - loss: 0.0225 - accuracy: 0.9893\n",
            "Epoch 66/70\n",
            "17/17 [==============================] - 1s 67ms/step - loss: 0.0179 - accuracy: 0.9952\n",
            "Epoch 67/70\n",
            "17/17 [==============================] - 1s 67ms/step - loss: 0.0224 - accuracy: 0.9884\n",
            "Epoch 68/70\n",
            "17/17 [==============================] - 1s 66ms/step - loss: 0.0227 - accuracy: 0.9913\n",
            "Epoch 69/70\n",
            "17/17 [==============================] - 1s 65ms/step - loss: 0.0586 - accuracy: 0.9748\n",
            "Epoch 70/70\n",
            "17/17 [==============================] - 1s 65ms/step - loss: 0.0340 - accuracy: 0.9874\n"
          ]
        },
        {
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7da307a4f6d0>"
            ]
          },
          "execution_count": 37,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "cnn.fit(X_train, y_train, epochs = 70, batch_size= 64)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yAngrOTfsfxd",
        "outputId": "b8b26278-9903-4a3e-98af-663a092415c7"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "9/9 [==============================] - 0s 8ms/step\n"
          ]
        }
      ],
      "source": [
        "pred = cnn.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "iL4r9DEbsiSV",
        "outputId": "6a4a5989-87fe-4955-c714-f4bcbedff590"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "(0.937984496124031,\n",
              " 0.9836065573770492,\n",
              " 0.8955223880597015,\n",
              " 0.8762293115855121,\n",
              " 0.8800291385253443)"
            ]
          },
          "execution_count": 39,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xzJJ07znsniP",
        "outputId": "04649b94-09d2-460a-eb1f-b9539b2d4c78"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "0.8970588235294118"
            ]
          },
          "execution_count": 40,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "oVYoQE_LiUc7"
      },
      "source": [
        "# **LSTM(LSA)**"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "JQYDUHzcipia"
      },
      "source": [
        "**Imbalanced**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "4OPX9GxJiYlj"
      },
      "outputs": [],
      "source": [
        "df1 = pd.read_csv('/content/LSA_TR.csv')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "H5BcPSSnr-mx"
      },
      "outputs": [],
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "yjZYoLD1sD1I"
      },
      "outputs": [],
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "iaJaL5DDsNMZ"
      },
      "outputs": [],
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "nkT0JT5Nscay"
      },
      "outputs": [],
      "source": [
        "model = Sequential()\n",
        "model.add(LSTM(units=256, input_shape=(X_train.shape[1], 1), return_sequences= True, activation = 'relu'))\n",
        "model.add(LSTM(256,return_sequences= True,  activation = 'relu'))\n",
        "model.add(LSTM(128,return_sequences= True,  activation = 'relu'))\n",
        "model.add(LSTM(64,return_sequences= True,  activation = 'relu'))\n",
        "model.add(Dense(units=1, activation='sigmoid'))  # Replace '1' with the number of output classes if you have a classification task\n",
        "model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yB9bnCPwsmQL",
        "outputId": "41f43df4-8591-4ecb-bbd0-17ce68d73430"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "41/41 [==============================] - 50s 1s/step - loss: 0.5933 - accuracy: 0.8028\n",
            "Epoch 2/75\n",
            "41/41 [==============================] - 37s 888ms/step - loss: 0.5198 - accuracy: 0.8029\n",
            "Epoch 3/75\n",
            "41/41 [==============================] - 32s 795ms/step - loss: 0.5090 - accuracy: 0.8029\n",
            "Epoch 4/75\n",
            "41/41 [==============================] - 32s 784ms/step - loss: 0.4953 - accuracy: 0.8027\n",
            "Epoch 5/75\n",
            "41/41 [==============================] - 33s 817ms/step - loss: 0.4731 - accuracy: 0.8058\n",
            "Epoch 6/75\n",
            "41/41 [==============================] - 32s 783ms/step - loss: 0.4445 - accuracy: 0.8137\n",
            "Epoch 7/75\n",
            "41/41 [==============================] - 34s 823ms/step - loss: 0.4283 - accuracy: 0.8165\n",
            "Epoch 8/75\n",
            "41/41 [==============================] - 34s 820ms/step - loss: 0.3927 - accuracy: 0.8319\n",
            "Epoch 9/75\n",
            "41/41 [==============================] - 36s 888ms/step - loss: 0.3730 - accuracy: 0.8395\n",
            "Epoch 10/75\n",
            "41/41 [==============================] - 32s 776ms/step - loss: 0.3453 - accuracy: 0.8488\n",
            "Epoch 11/75\n",
            "41/41 [==============================] - 36s 869ms/step - loss: 0.3258 - accuracy: 0.8525\n",
            "Epoch 12/75\n",
            "41/41 [==============================] - 32s 782ms/step - loss: 0.3165 - accuracy: 0.8592\n",
            "Epoch 13/75\n",
            "41/41 [==============================] - 33s 803ms/step - loss: 0.3048 - accuracy: 0.8659\n",
            "Epoch 14/75\n",
            "41/41 [==============================] - 35s 853ms/step - loss: 0.2965 - accuracy: 0.8703\n",
            "Epoch 15/75\n",
            "41/41 [==============================] - 34s 840ms/step - loss: 0.2908 - accuracy: 0.8762\n",
            "Epoch 16/75\n",
            "41/41 [==============================] - 33s 803ms/step - loss: 0.3028 - accuracy: 0.8682\n",
            "Epoch 17/75\n",
            "41/41 [==============================] - 34s 821ms/step - loss: 0.3019 - accuracy: 0.8721\n",
            "Epoch 18/75\n",
            "41/41 [==============================] - 34s 844ms/step - loss: 0.2763 - accuracy: 0.8840\n",
            "Epoch 19/75\n",
            "41/41 [==============================] - 41s 988ms/step - loss: 0.2682 - accuracy: 0.8849\n",
            "Epoch 20/75\n",
            "41/41 [==============================] - 34s 841ms/step - loss: 0.2728 - accuracy: 0.8843\n",
            "Epoch 21/75\n",
            "41/41 [==============================] - 34s 821ms/step - loss: 0.2736 - accuracy: 0.8875\n",
            "Epoch 22/75\n",
            "41/41 [==============================] - 34s 822ms/step - loss: 0.2612 - accuracy: 0.8913\n",
            "Epoch 23/75\n",
            "41/41 [==============================] - 36s 874ms/step - loss: 0.2584 - accuracy: 0.8926\n",
            "Epoch 24/75\n",
            "41/41 [==============================] - 35s 837ms/step - loss: 0.2534 - accuracy: 0.8949\n",
            "Epoch 25/75\n",
            "41/41 [==============================] - 34s 838ms/step - loss: 0.2602 - accuracy: 0.8929\n",
            "Epoch 26/75\n",
            "41/41 [==============================] - 35s 847ms/step - loss: 0.2436 - accuracy: 0.8979\n",
            "Epoch 27/75\n",
            "41/41 [==============================] - 34s 820ms/step - loss: 0.2300 - accuracy: 0.9061\n",
            "Epoch 28/75\n",
            "41/41 [==============================] - 34s 837ms/step - loss: 0.2781 - accuracy: 0.8858\n",
            "Epoch 29/75\n",
            "41/41 [==============================] - 34s 832ms/step - loss: 0.2499 - accuracy: 0.8958\n",
            "Epoch 30/75\n",
            "41/41 [==============================] - 33s 796ms/step - loss: 0.2297 - accuracy: 0.9060\n",
            "Epoch 31/75\n",
            "41/41 [==============================] - 33s 819ms/step - loss: 0.2306 - accuracy: 0.9063\n",
            "Epoch 32/75\n",
            "41/41 [==============================] - 32s 782ms/step - loss: 0.2164 - accuracy: 0.9123\n",
            "Epoch 33/75\n",
            "41/41 [==============================] - 33s 787ms/step - loss: 0.2125 - accuracy: 0.9145\n",
            "Epoch 34/75\n",
            "41/41 [==============================] - 33s 815ms/step - loss: 0.2190 - accuracy: 0.9140\n",
            "Epoch 35/75\n",
            "41/41 [==============================] - 32s 792ms/step - loss: 0.2076 - accuracy: 0.9167\n",
            "Epoch 36/75\n",
            "41/41 [==============================] - 33s 811ms/step - loss: 0.2122 - accuracy: 0.9155\n",
            "Epoch 37/75\n",
            "41/41 [==============================] - 33s 796ms/step - loss: 0.2123 - accuracy: 0.9171\n",
            "Epoch 38/75\n",
            "41/41 [==============================] - 34s 817ms/step - loss: 0.2060 - accuracy: 0.9176\n",
            "Epoch 39/75\n",
            "41/41 [==============================] - 33s 806ms/step - loss: 0.2023 - accuracy: 0.9188\n",
            "Epoch 40/75\n",
            "41/41 [==============================] - 32s 776ms/step - loss: 0.1975 - accuracy: 0.9212\n",
            "Epoch 41/75\n",
            "41/41 [==============================] - 33s 802ms/step - loss: 0.1997 - accuracy: 0.9209\n",
            "Epoch 42/75\n",
            "41/41 [==============================] - 32s 775ms/step - loss: 0.1888 - accuracy: 0.9270\n",
            "Epoch 43/75\n",
            "41/41 [==============================] - 33s 794ms/step - loss: 0.1952 - accuracy: 0.9237\n",
            "Epoch 44/75\n",
            "41/41 [==============================] - 32s 784ms/step - loss: 0.2093 - accuracy: 0.9192\n",
            "Epoch 45/75\n",
            "41/41 [==============================] - 32s 776ms/step - loss: 0.2022 - accuracy: 0.9184\n",
            "Epoch 46/75\n",
            "41/41 [==============================] - 33s 798ms/step - loss: 0.2089 - accuracy: 0.9130\n",
            "Epoch 47/75\n",
            "41/41 [==============================] - 32s 765ms/step - loss: 0.2021 - accuracy: 0.9204\n",
            "Epoch 48/75\n",
            "41/41 [==============================] - 32s 793ms/step - loss: 0.1788 - accuracy: 0.9300\n",
            "Epoch 49/75\n",
            "41/41 [==============================] - 32s 779ms/step - loss: 0.1832 - accuracy: 0.9294\n",
            "Epoch 50/75\n",
            "41/41 [==============================] - 33s 798ms/step - loss: 0.1771 - accuracy: 0.9295\n",
            "Epoch 51/75\n",
            "41/41 [==============================] - 31s 767ms/step - loss: 0.1688 - accuracy: 0.9355\n",
            "Epoch 52/75\n",
            "41/41 [==============================] - 32s 791ms/step - loss: 0.1684 - accuracy: 0.9339\n",
            "Epoch 53/75\n",
            "41/41 [==============================] - 31s 769ms/step - loss: 0.1830 - accuracy: 0.9269\n",
            "Epoch 54/75\n",
            "41/41 [==============================] - 33s 806ms/step - loss: 0.1876 - accuracy: 0.9307\n",
            "Epoch 55/75\n",
            "41/41 [==============================] - 32s 785ms/step - loss: 0.1898 - accuracy: 0.9243\n",
            "Epoch 56/75\n",
            "41/41 [==============================] - 33s 800ms/step - loss: 0.1709 - accuracy: 0.9363\n",
            "Epoch 57/75\n",
            "41/41 [==============================] - 38s 934ms/step - loss: 0.1521 - accuracy: 0.9399\n",
            "Epoch 58/75\n",
            "41/41 [==============================] - 34s 826ms/step - loss: 0.1573 - accuracy: 0.9397\n",
            "Epoch 59/75\n",
            "41/41 [==============================] - 33s 808ms/step - loss: 0.1578 - accuracy: 0.9400\n",
            "Epoch 60/75\n",
            "41/41 [==============================] - 34s 832ms/step - loss: 0.1537 - accuracy: 0.9401\n",
            "Epoch 61/75\n",
            "41/41 [==============================] - 35s 853ms/step - loss: 0.1565 - accuracy: 0.9388\n",
            "Epoch 62/75\n",
            "41/41 [==============================] - 34s 836ms/step - loss: 0.1637 - accuracy: 0.9363\n",
            "Epoch 63/75\n",
            "41/41 [==============================] - 33s 805ms/step - loss: 0.1505 - accuracy: 0.9422\n",
            "Epoch 64/75\n",
            "41/41 [==============================] - 33s 799ms/step - loss: 0.2252 - accuracy: 0.9213\n",
            "Epoch 65/75\n",
            "41/41 [==============================] - 34s 830ms/step - loss: 0.2785 - accuracy: 0.8859\n",
            "Epoch 66/75\n",
            "41/41 [==============================] - 35s 829ms/step - loss: 0.2122 - accuracy: 0.9184\n",
            "Epoch 67/75\n",
            "41/41 [==============================] - 32s 792ms/step - loss: 0.1882 - accuracy: 0.9260\n",
            "Epoch 68/75\n",
            "41/41 [==============================] - 33s 787ms/step - loss: 0.1754 - accuracy: 0.9313\n",
            "Epoch 69/75\n",
            "41/41 [==============================] - 34s 821ms/step - loss: 0.1818 - accuracy: 0.9285\n",
            "Epoch 70/75\n",
            "41/41 [==============================] - 37s 905ms/step - loss: 0.1597 - accuracy: 0.9380\n",
            "Epoch 71/75\n",
            "41/41 [==============================] - 32s 781ms/step - loss: 0.1889 - accuracy: 0.9239\n",
            "Epoch 72/75\n",
            "41/41 [==============================] - 33s 810ms/step - loss: 0.1571 - accuracy: 0.9371\n",
            "Epoch 73/75\n",
            "41/41 [==============================] - 32s 791ms/step - loss: 0.1575 - accuracy: 0.9409\n",
            "Epoch 74/75\n",
            "41/41 [==============================] - 36s 893ms/step - loss: 0.1478 - accuracy: 0.9412\n",
            "Epoch 75/75\n",
            "41/41 [==============================] - 33s 811ms/step - loss: 0.1440 - accuracy: 0.9449\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x794ccd79ec20>"
            ]
          },
          "metadata": {},
          "execution_count": 23
        }
      ],
      "source": [
        "model.fit(X_train, y_train, epochs=75, batch_size=64)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "dsvE-fEttDFo",
        "outputId": "342a4642-87b4-4623-e08b-78d2e2c000ea"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "21/21 [==============================] - 0s 18ms/step\n"
          ]
        }
      ],
      "source": [
        "pred = cnn.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "sQXUj_VlzWvo",
        "outputId": "fab8a701-e850-47d5-9744-d2dcea655668"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.8480620155038759,\n",
              " 1.0,\n",
              " 0.5811965811965812,\n",
              " 0.6388041279528234,\n",
              " 0.6850520105185522)"
            ]
          },
          "metadata": {},
          "execution_count": 25
        }
      ],
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "TcrkK5g-tG1x",
        "outputId": "425c827e-24fd-4598-b8c0-42bca3499cba"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.8074656188605108"
            ]
          },
          "metadata": {},
          "execution_count": 26
        }
      ],
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "o-o-yNFvzoKQ"
      },
      "source": [
        "# **CNN(LSA)**"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Imbalanced**"
      ],
      "metadata": {
        "id": "gRF5K4RfpUMx"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "oEsafzXsztZ4"
      },
      "outputs": [],
      "source": [
        "df1 = pd.read_csv('/content/LSA_TR.csv')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "tO4sSMnTz5-J"
      },
      "outputs": [],
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "wXbLHTRrz8li"
      },
      "outputs": [],
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "L49ws_pJz-5J"
      },
      "outputs": [],
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "-LLEhPFL0BTx"
      },
      "outputs": [],
      "source": [
        "cnn = Sequential()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "yHuUJCjm0DQI"
      },
      "outputs": [],
      "source": [
        "cnn.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))\n",
        "cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))\n",
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu'))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "6IfoSwgN0K5x"
      },
      "outputs": [],
      "source": [
        "cnn.add(MaxPool1D(pool_size=2))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Crc09oAu0aMY"
      },
      "outputs": [],
      "source": [
        "cnn.add(Flatten())"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "4XlHLl7d0gdj"
      },
      "outputs": [],
      "source": [
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "4rubjWvv0hTL"
      },
      "outputs": [],
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "9VEDYc3L0jUh",
        "outputId": "51caa2bd-d43b-4476-c589-e31f90a58777"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "41/41 [==============================] - 5s 86ms/step - loss: 0.5231 - accuracy: 0.7932\n",
            "Epoch 2/75\n",
            "41/41 [==============================] - 4s 106ms/step - loss: 0.4710 - accuracy: 0.8033\n",
            "Epoch 3/75\n",
            "41/41 [==============================] - 4s 101ms/step - loss: 0.4076 - accuracy: 0.8033\n",
            "Epoch 4/75\n",
            "41/41 [==============================] - 4s 99ms/step - loss: 0.3725 - accuracy: 0.8033\n",
            "Epoch 5/75\n",
            "41/41 [==============================] - 3s 74ms/step - loss: 0.2756 - accuracy: 0.8737\n",
            "Epoch 6/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.2217 - accuracy: 0.9167\n",
            "Epoch 7/75\n",
            "41/41 [==============================] - 2s 47ms/step - loss: 0.1996 - accuracy: 0.9369\n",
            "Epoch 8/75\n",
            "41/41 [==============================] - 2s 46ms/step - loss: 0.1907 - accuracy: 0.9419\n",
            "Epoch 9/75\n",
            "41/41 [==============================] - 3s 67ms/step - loss: 0.1782 - accuracy: 0.9524\n",
            "Epoch 10/75\n",
            "41/41 [==============================] - 2s 46ms/step - loss: 0.1528 - accuracy: 0.9706\n",
            "Epoch 11/75\n",
            "41/41 [==============================] - 2s 46ms/step - loss: 0.1395 - accuracy: 0.9783\n",
            "Epoch 12/75\n",
            "41/41 [==============================] - 2s 45ms/step - loss: 0.1303 - accuracy: 0.9849\n",
            "Epoch 13/75\n",
            "41/41 [==============================] - 2s 45ms/step - loss: 0.1230 - accuracy: 0.9899\n",
            "Epoch 14/75\n",
            "41/41 [==============================] - 2s 44ms/step - loss: 0.1169 - accuracy: 0.9923\n",
            "Epoch 15/75\n",
            "41/41 [==============================] - 3s 63ms/step - loss: 0.1121 - accuracy: 0.9957\n",
            "Epoch 16/75\n",
            "41/41 [==============================] - 2s 44ms/step - loss: 0.1077 - accuracy: 0.9961\n",
            "Epoch 17/75\n",
            "41/41 [==============================] - 2s 45ms/step - loss: 0.1039 - accuracy: 0.9969\n",
            "Epoch 18/75\n",
            "41/41 [==============================] - 2s 47ms/step - loss: 0.0998 - accuracy: 0.9973\n",
            "Epoch 19/75\n",
            "41/41 [==============================] - 2s 46ms/step - loss: 0.0970 - accuracy: 0.9973\n",
            "Epoch 20/75\n",
            "41/41 [==============================] - 2s 46ms/step - loss: 0.0939 - accuracy: 0.9973\n",
            "Epoch 21/75\n",
            "41/41 [==============================] - 3s 63ms/step - loss: 0.0908 - accuracy: 0.9973\n",
            "Epoch 22/75\n",
            "41/41 [==============================] - 2s 47ms/step - loss: 0.0880 - accuracy: 0.9973\n",
            "Epoch 23/75\n",
            "41/41 [==============================] - 2s 46ms/step - loss: 0.0857 - accuracy: 0.9977\n",
            "Epoch 24/75\n",
            "41/41 [==============================] - 2s 49ms/step - loss: 0.0831 - accuracy: 0.9977\n",
            "Epoch 25/75\n",
            "41/41 [==============================] - 2s 49ms/step - loss: 0.0808 - accuracy: 0.9981\n",
            "Epoch 26/75\n",
            "41/41 [==============================] - 2s 47ms/step - loss: 0.0784 - accuracy: 0.9985\n",
            "Epoch 27/75\n",
            "41/41 [==============================] - 3s 66ms/step - loss: 0.0763 - accuracy: 0.9985\n",
            "Epoch 28/75\n",
            "41/41 [==============================] - 2s 46ms/step - loss: 0.0740 - accuracy: 0.9985\n",
            "Epoch 29/75\n",
            "41/41 [==============================] - 2s 45ms/step - loss: 0.0719 - accuracy: 0.9985\n",
            "Epoch 30/75\n",
            "41/41 [==============================] - 2s 47ms/step - loss: 0.0701 - accuracy: 0.9985\n",
            "Epoch 31/75\n",
            "41/41 [==============================] - 2s 45ms/step - loss: 0.0682 - accuracy: 0.9985\n",
            "Epoch 32/75\n",
            "41/41 [==============================] - 2s 45ms/step - loss: 0.0665 - accuracy: 0.9985\n",
            "Epoch 33/75\n",
            "41/41 [==============================] - 3s 63ms/step - loss: 0.0649 - accuracy: 0.9985\n",
            "Epoch 34/75\n",
            "41/41 [==============================] - 2s 46ms/step - loss: 0.0632 - accuracy: 0.9985\n",
            "Epoch 35/75\n",
            "41/41 [==============================] - 2s 46ms/step - loss: 0.0615 - accuracy: 0.9985\n",
            "Epoch 36/75\n",
            "41/41 [==============================] - 2s 45ms/step - loss: 0.0600 - accuracy: 0.9985\n",
            "Epoch 37/75\n",
            "41/41 [==============================] - 2s 46ms/step - loss: 0.0586 - accuracy: 0.9985\n",
            "Epoch 38/75\n",
            "41/41 [==============================] - 2s 44ms/step - loss: 0.0572 - accuracy: 0.9985\n",
            "Epoch 39/75\n",
            "41/41 [==============================] - 3s 73ms/step - loss: 0.0558 - accuracy: 0.9985\n",
            "Epoch 40/75\n",
            "41/41 [==============================] - 2s 46ms/step - loss: 0.0545 - accuracy: 0.9985\n",
            "Epoch 41/75\n",
            "41/41 [==============================] - 2s 47ms/step - loss: 0.0533 - accuracy: 0.9985\n",
            "Epoch 42/75\n",
            "41/41 [==============================] - 2s 47ms/step - loss: 0.0520 - accuracy: 0.9985\n",
            "Epoch 43/75\n",
            "41/41 [==============================] - 2s 46ms/step - loss: 0.0509 - accuracy: 0.9985\n",
            "Epoch 44/75\n",
            "41/41 [==============================] - 2s 45ms/step - loss: 0.0497 - accuracy: 0.9985\n",
            "Epoch 45/75\n",
            "41/41 [==============================] - 3s 63ms/step - loss: 0.0486 - accuracy: 0.9985\n",
            "Epoch 46/75\n",
            "41/41 [==============================] - 2s 45ms/step - loss: 0.0475 - accuracy: 0.9985\n",
            "Epoch 47/75\n",
            "41/41 [==============================] - 2s 45ms/step - loss: 0.0465 - accuracy: 0.9985\n",
            "Epoch 48/75\n",
            "41/41 [==============================] - 2s 45ms/step - loss: 0.0455 - accuracy: 0.9985\n",
            "Epoch 49/75\n",
            "41/41 [==============================] - 2s 47ms/step - loss: 0.0445 - accuracy: 0.9985\n",
            "Epoch 50/75\n",
            "41/41 [==============================] - 2s 47ms/step - loss: 0.0435 - accuracy: 0.9985\n",
            "Epoch 51/75\n",
            "41/41 [==============================] - 3s 64ms/step - loss: 0.0426 - accuracy: 0.9985\n",
            "Epoch 52/75\n",
            "41/41 [==============================] - 2s 46ms/step - loss: 0.0417 - accuracy: 0.9985\n",
            "Epoch 53/75\n",
            "41/41 [==============================] - 2s 47ms/step - loss: 0.0409 - accuracy: 0.9985\n",
            "Epoch 54/75\n",
            "41/41 [==============================] - 2s 47ms/step - loss: 0.0401 - accuracy: 0.9985\n",
            "Epoch 55/75\n",
            "41/41 [==============================] - 2s 47ms/step - loss: 0.0392 - accuracy: 0.9985\n",
            "Epoch 56/75\n",
            "41/41 [==============================] - 2s 47ms/step - loss: 0.0385 - accuracy: 0.9985\n",
            "Epoch 57/75\n",
            "41/41 [==============================] - 3s 63ms/step - loss: 0.0377 - accuracy: 0.9985\n",
            "Epoch 58/75\n",
            "41/41 [==============================] - 2s 45ms/step - loss: 0.0370 - accuracy: 0.9985\n",
            "Epoch 59/75\n",
            "41/41 [==============================] - 2s 46ms/step - loss: 0.0362 - accuracy: 0.9985\n",
            "Epoch 60/75\n",
            "41/41 [==============================] - 2s 45ms/step - loss: 0.0355 - accuracy: 0.9985\n",
            "Epoch 61/75\n",
            "41/41 [==============================] - 2s 46ms/step - loss: 0.0349 - accuracy: 0.9985\n",
            "Epoch 62/75\n",
            "41/41 [==============================] - 2s 46ms/step - loss: 0.0342 - accuracy: 0.9985\n",
            "Epoch 63/75\n",
            "41/41 [==============================] - 3s 63ms/step - loss: 0.0336 - accuracy: 0.9985\n",
            "Epoch 64/75\n",
            "41/41 [==============================] - 2s 47ms/step - loss: 0.0329 - accuracy: 0.9985\n",
            "Epoch 65/75\n",
            "41/41 [==============================] - 2s 46ms/step - loss: 0.0323 - accuracy: 0.9985\n",
            "Epoch 66/75\n",
            "41/41 [==============================] - 2s 46ms/step - loss: 0.0317 - accuracy: 0.9985\n",
            "Epoch 67/75\n",
            "41/41 [==============================] - 2s 45ms/step - loss: 0.0312 - accuracy: 0.9985\n",
            "Epoch 68/75\n",
            "41/41 [==============================] - 2s 47ms/step - loss: 0.0306 - accuracy: 0.9985\n",
            "Epoch 69/75\n",
            "41/41 [==============================] - 3s 62ms/step - loss: 0.0300 - accuracy: 0.9985\n",
            "Epoch 70/75\n",
            "41/41 [==============================] - 2s 45ms/step - loss: 0.0295 - accuracy: 0.9985\n",
            "Epoch 71/75\n",
            "41/41 [==============================] - 2s 46ms/step - loss: 0.0290 - accuracy: 0.9985\n",
            "Epoch 72/75\n",
            "41/41 [==============================] - 2s 47ms/step - loss: 0.0285 - accuracy: 0.9985\n",
            "Epoch 73/75\n",
            "41/41 [==============================] - 2s 45ms/step - loss: 0.0280 - accuracy: 0.9985\n",
            "Epoch 74/75\n",
            "41/41 [==============================] - 2s 46ms/step - loss: 0.0275 - accuracy: 0.9985\n",
            "Epoch 75/75\n",
            "41/41 [==============================] - 3s 70ms/step - loss: 0.0271 - accuracy: 0.9985\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7967c4cc9ae0>"
            ]
          },
          "metadata": {},
          "execution_count": 15
        }
      ],
      "source": [
        "cnn.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cG974VaY2g9x",
        "outputId": "4204fd65-4ff7-45e3-fd3b-7818dcba1c38"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "21/21 [==============================] - 0s 8ms/step\n"
          ]
        }
      ],
      "source": [
        "pred = cnn.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "NBsfcEgw2jtR",
        "outputId": "5213c752-e396-4571-d909-f14efe9284d0"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9875968992248062,\n",
              " 1.0,\n",
              " 0.9448275862068966,\n",
              " 0.9637028700056275,\n",
              " 0.9643383283998211)"
            ]
          },
          "metadata": {},
          "execution_count": 17
        }
      ],
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Dqj8sUQw2n0C",
        "outputId": "28254831-798d-4776-8412-d01c70ed709f"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.984251968503937"
            ]
          },
          "metadata": {},
          "execution_count": 18
        }
      ],
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Test**"
      ],
      "metadata": {
        "id": "3O4fYp_6pcSX"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/LSA_TR.csv')\n",
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "Bz_bId14peH3"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.3, random_state = 1)"
      ],
      "metadata": {
        "id": "o0Jywy-2nOx5"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "sample_size = xtrain.shape[0] # number of samples in train set\n",
        "time_steps  = xtrain.shape[1] # number of features in train set\n",
        "input_dimension = 1               # each feature is represented by 1 number"
      ],
      "metadata": {
        "id": "ZeI2Fv12nENt"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_data_reshaped = xtrain.values.reshape(sample_size,time_steps,input_dimension)\n",
        "n_timesteps = train_data_reshaped.shape[1]\n",
        "n_features  = train_data_reshaped.shape[2]"
      ],
      "metadata": {
        "id": "IDbX7_ZxnHLc"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn = Sequential()"
      ],
      "metadata": {
        "id": "tqUkwUPInJKU"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))\n",
        "cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))\n",
        "cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))"
      ],
      "metadata": {
        "id": "D12Omp26nLtQ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "zLzqXqVenNy8"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Flatten())"
      ],
      "metadata": {
        "id": "etJCxuz8nPtj"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "8KWKCFhunRrg"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "H0WlF9kUnT1W"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.fit(xtrain, ytrain, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "OaxeR12KnWMn",
        "outputId": "e11caf9b-2c48-4e18-9880-29ba14a600f1"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "36/36 [==============================] - 3s 53ms/step - loss: 0.5229 - accuracy: 0.8091\n",
            "Epoch 2/75\n",
            "36/36 [==============================] - 2s 59ms/step - loss: 0.4731 - accuracy: 0.8127\n",
            "Epoch 3/75\n",
            "36/36 [==============================] - 2s 44ms/step - loss: 0.4054 - accuracy: 0.8193\n",
            "Epoch 4/75\n",
            "36/36 [==============================] - 1s 41ms/step - loss: 0.3672 - accuracy: 0.8477\n",
            "Epoch 5/75\n",
            "36/36 [==============================] - 1s 31ms/step - loss: 0.3604 - accuracy: 0.8516\n",
            "Epoch 6/75\n",
            "36/36 [==============================] - 1s 29ms/step - loss: 0.2941 - accuracy: 0.8729\n",
            "Epoch 7/75\n",
            "36/36 [==============================] - 1s 30ms/step - loss: 0.2312 - accuracy: 0.8981\n",
            "Epoch 8/75\n",
            "36/36 [==============================] - 1s 30ms/step - loss: 0.1685 - accuracy: 0.9322\n",
            "Epoch 9/75\n",
            "36/36 [==============================] - 1s 30ms/step - loss: 0.1281 - accuracy: 0.9526\n",
            "Epoch 10/75\n",
            "36/36 [==============================] - 1s 39ms/step - loss: 0.0943 - accuracy: 0.9690\n",
            "Epoch 11/75\n",
            "36/36 [==============================] - 2s 43ms/step - loss: 0.0848 - accuracy: 0.9770\n",
            "Epoch 12/75\n",
            "36/36 [==============================] - 1s 31ms/step - loss: 0.0573 - accuracy: 0.9880\n",
            "Epoch 13/75\n",
            "36/36 [==============================] - 1s 32ms/step - loss: 0.0455 - accuracy: 0.9894\n",
            "Epoch 14/75\n",
            "36/36 [==============================] - 1s 31ms/step - loss: 0.0342 - accuracy: 0.9920\n",
            "Epoch 15/75\n",
            "36/36 [==============================] - 1s 30ms/step - loss: 0.0321 - accuracy: 0.9925\n",
            "Epoch 16/75\n",
            "36/36 [==============================] - 1s 31ms/step - loss: 0.0285 - accuracy: 0.9934\n",
            "Epoch 17/75\n",
            "36/36 [==============================] - 1s 30ms/step - loss: 0.0386 - accuracy: 0.9880\n",
            "Epoch 18/75\n",
            "36/36 [==============================] - 1s 30ms/step - loss: 0.0347 - accuracy: 0.9880\n",
            "Epoch 19/75\n",
            "36/36 [==============================] - 1s 34ms/step - loss: 0.0143 - accuracy: 0.9987\n",
            "Epoch 20/75\n",
            "36/36 [==============================] - 1s 40ms/step - loss: 0.0093 - accuracy: 0.9987\n",
            "Epoch 21/75\n",
            "36/36 [==============================] - 2s 45ms/step - loss: 0.0077 - accuracy: 0.9982\n",
            "Epoch 22/75\n",
            "36/36 [==============================] - 1s 33ms/step - loss: 0.0058 - accuracy: 0.9991\n",
            "Epoch 23/75\n",
            "36/36 [==============================] - 1s 32ms/step - loss: 0.0046 - accuracy: 0.9991\n",
            "Epoch 24/75\n",
            "36/36 [==============================] - 1s 30ms/step - loss: 0.0037 - accuracy: 0.9991\n",
            "Epoch 25/75\n",
            "36/36 [==============================] - 1s 30ms/step - loss: 0.0030 - accuracy: 0.9991\n",
            "Epoch 26/75\n",
            "36/36 [==============================] - 1s 32ms/step - loss: 0.0026 - accuracy: 1.0000\n",
            "Epoch 27/75\n",
            "36/36 [==============================] - 1s 31ms/step - loss: 0.0022 - accuracy: 0.9996\n",
            "Epoch 28/75\n",
            "36/36 [==============================] - 1s 31ms/step - loss: 0.0017 - accuracy: 1.0000\n",
            "Epoch 29/75\n",
            "36/36 [==============================] - 1s 30ms/step - loss: 0.0015 - accuracy: 1.0000\n",
            "Epoch 30/75\n",
            "36/36 [==============================] - 1s 39ms/step - loss: 0.0013 - accuracy: 1.0000\n",
            "Epoch 31/75\n",
            "36/36 [==============================] - 2s 43ms/step - loss: 0.0011 - accuracy: 1.0000\n",
            "Epoch 32/75\n",
            "36/36 [==============================] - 1s 30ms/step - loss: 9.6272e-04 - accuracy: 1.0000\n",
            "Epoch 33/75\n",
            "36/36 [==============================] - 1s 30ms/step - loss: 8.7184e-04 - accuracy: 1.0000\n",
            "Epoch 34/75\n",
            "36/36 [==============================] - 1s 31ms/step - loss: 7.7630e-04 - accuracy: 1.0000\n",
            "Epoch 35/75\n",
            "36/36 [==============================] - 1s 35ms/step - loss: 7.2413e-04 - accuracy: 1.0000\n",
            "Epoch 36/75\n",
            "36/36 [==============================] - 1s 31ms/step - loss: 6.7696e-04 - accuracy: 1.0000\n",
            "Epoch 37/75\n",
            "36/36 [==============================] - 1s 31ms/step - loss: 5.9004e-04 - accuracy: 1.0000\n",
            "Epoch 38/75\n",
            "36/36 [==============================] - 1s 34ms/step - loss: 5.4372e-04 - accuracy: 1.0000\n",
            "Epoch 39/75\n",
            "36/36 [==============================] - 1s 32ms/step - loss: 5.0123e-04 - accuracy: 1.0000\n",
            "Epoch 40/75\n",
            "36/36 [==============================] - 2s 43ms/step - loss: 4.7561e-04 - accuracy: 1.0000\n",
            "Epoch 41/75\n",
            "36/36 [==============================] - 1s 40ms/step - loss: 4.2566e-04 - accuracy: 1.0000\n",
            "Epoch 42/75\n",
            "36/36 [==============================] - 1s 31ms/step - loss: 4.0917e-04 - accuracy: 1.0000\n",
            "Epoch 43/75\n",
            "36/36 [==============================] - 1s 30ms/step - loss: 3.7815e-04 - accuracy: 1.0000\n",
            "Epoch 44/75\n",
            "36/36 [==============================] - 1s 29ms/step - loss: 3.5340e-04 - accuracy: 1.0000\n",
            "Epoch 45/75\n",
            "36/36 [==============================] - 1s 31ms/step - loss: 3.2473e-04 - accuracy: 1.0000\n",
            "Epoch 46/75\n",
            "36/36 [==============================] - 1s 34ms/step - loss: 3.1162e-04 - accuracy: 1.0000\n",
            "Epoch 47/75\n",
            "36/36 [==============================] - 1s 34ms/step - loss: 2.9091e-04 - accuracy: 1.0000\n",
            "Epoch 48/75\n",
            "36/36 [==============================] - 1s 31ms/step - loss: 2.6654e-04 - accuracy: 1.0000\n",
            "Epoch 49/75\n",
            "36/36 [==============================] - 1s 32ms/step - loss: 2.5494e-04 - accuracy: 1.0000\n",
            "Epoch 50/75\n",
            "36/36 [==============================] - 2s 42ms/step - loss: 2.3846e-04 - accuracy: 1.0000\n",
            "Epoch 51/75\n",
            "36/36 [==============================] - 1s 41ms/step - loss: 2.2567e-04 - accuracy: 1.0000\n",
            "Epoch 52/75\n",
            "36/36 [==============================] - 1s 34ms/step - loss: 2.1334e-04 - accuracy: 1.0000\n",
            "Epoch 53/75\n",
            "36/36 [==============================] - 1s 30ms/step - loss: 2.0430e-04 - accuracy: 1.0000\n",
            "Epoch 54/75\n",
            "36/36 [==============================] - 1s 32ms/step - loss: 1.9017e-04 - accuracy: 1.0000\n",
            "Epoch 55/75\n",
            "36/36 [==============================] - 1s 31ms/step - loss: 1.8352e-04 - accuracy: 1.0000\n",
            "Epoch 56/75\n",
            "36/36 [==============================] - 1s 31ms/step - loss: 1.7325e-04 - accuracy: 1.0000\n",
            "Epoch 57/75\n",
            "36/36 [==============================] - 1s 34ms/step - loss: 1.6515e-04 - accuracy: 1.0000\n",
            "Epoch 58/75\n",
            "36/36 [==============================] - 1s 33ms/step - loss: 1.5731e-04 - accuracy: 1.0000\n",
            "Epoch 59/75\n",
            "36/36 [==============================] - 1s 33ms/step - loss: 1.4948e-04 - accuracy: 1.0000\n",
            "Epoch 60/75\n",
            "36/36 [==============================] - 2s 51ms/step - loss: 1.4255e-04 - accuracy: 1.0000\n",
            "Epoch 61/75\n",
            "36/36 [==============================] - 1s 33ms/step - loss: 1.3722e-04 - accuracy: 1.0000\n",
            "Epoch 62/75\n",
            "36/36 [==============================] - 1s 32ms/step - loss: 1.3214e-04 - accuracy: 1.0000\n",
            "Epoch 63/75\n",
            "36/36 [==============================] - 1s 32ms/step - loss: 1.2938e-04 - accuracy: 1.0000\n",
            "Epoch 64/75\n",
            "36/36 [==============================] - 1s 31ms/step - loss: 1.2061e-04 - accuracy: 1.0000\n",
            "Epoch 65/75\n",
            "36/36 [==============================] - 1s 30ms/step - loss: 1.1604e-04 - accuracy: 1.0000\n",
            "Epoch 66/75\n",
            "36/36 [==============================] - 1s 30ms/step - loss: 1.0912e-04 - accuracy: 1.0000\n",
            "Epoch 67/75\n",
            "36/36 [==============================] - 1s 31ms/step - loss: 1.0639e-04 - accuracy: 1.0000\n",
            "Epoch 68/75\n",
            "36/36 [==============================] - 1s 29ms/step - loss: 1.0036e-04 - accuracy: 1.0000\n",
            "Epoch 69/75\n",
            "36/36 [==============================] - 1s 30ms/step - loss: 9.6643e-05 - accuracy: 1.0000\n",
            "Epoch 70/75\n",
            "36/36 [==============================] - 2s 43ms/step - loss: 9.1880e-05 - accuracy: 1.0000\n",
            "Epoch 71/75\n",
            "36/36 [==============================] - 1s 39ms/step - loss: 8.9083e-05 - accuracy: 1.0000\n",
            "Epoch 72/75\n",
            "36/36 [==============================] - 1s 31ms/step - loss: 8.8968e-05 - accuracy: 1.0000\n",
            "Epoch 73/75\n",
            "36/36 [==============================] - 1s 32ms/step - loss: 8.2232e-05 - accuracy: 1.0000\n",
            "Epoch 74/75\n",
            "36/36 [==============================] - 1s 30ms/step - loss: 7.9548e-05 - accuracy: 1.0000\n",
            "Epoch 75/75\n",
            "36/36 [==============================] - 1s 30ms/step - loss: 7.6288e-05 - accuracy: 1.0000\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7967b651d570>"
            ]
          },
          "metadata": {},
          "execution_count": 38
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cnn.predict(xtest)\n",
        "pred = (pred > 0.5)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "pFKkUkHLnYID",
        "outputId": "87c72941-0af1-4264-d6cf-32dddf12d5f5"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "31/31 [==============================] - 0s 5ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(ytest, pred), precision_score(ytest, pred), recall_score(ytest, pred), f1_score(ytest, pred), cohen_kappa_score(ytest, pred), matthews_corrcoef(ytest, pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "eq93TWHyna69",
        "outputId": "d53db8b9-e6bd-4a6f-f64f-8d1103699268"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9731682146542827,\n",
              " 0.9188034188034188,\n",
              " 0.9684684684684685,\n",
              " 0.9429824561403509,\n",
              " 0.9254544808975997,\n",
              " 0.9260027196495192)"
            ]
          },
          "metadata": {},
          "execution_count": 40
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(ytest, pred)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "v2IEeyZDndKE",
        "outputId": "582b358c-8ff9-44fc-a2d3-6d85f94b8c9c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9745649263721553"
            ]
          },
          "metadata": {},
          "execution_count": 41
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "d8K8Q-oBZMrN"
      },
      "source": [
        "**ADASYN**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "JeZTMW0HZPR8"
      },
      "outputs": [],
      "source": [
        "df1 = pd.read_csv('/content/LSA_TR.csv')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "X1w2-wQ4Zfpl"
      },
      "outputs": [],
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "bAnXai7sZkBH"
      },
      "outputs": [],
      "source": [
        "from imblearn.over_sampling import ADASYN\n",
        "ada = ADASYN()\n",
        "X, Y = ada.fit_resample(X, Y)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "RvWuCQefZs6l"
      },
      "outputs": [],
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "HfN_7vOoZyZU"
      },
      "outputs": [],
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "IC-U8BBRZ4TV"
      },
      "outputs": [],
      "source": [
        "cnn = Sequential()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "WwflUWonZ9TM"
      },
      "outputs": [],
      "source": [
        "cnn.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))\n",
        "cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))\n",
        "cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "kyYscT-JaBdt"
      },
      "outputs": [],
      "source": [
        "cnn.add(MaxPool1D(pool_size=2))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "xXmE6lZzaEvd"
      },
      "outputs": [],
      "source": [
        "cnn.add(Flatten())"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "MXELd8rLaK9t"
      },
      "outputs": [],
      "source": [
        "cnn.add(Dense(128, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "gyT4P6XlaOjF"
      },
      "outputs": [],
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "HNMU5mezaRvF",
        "outputId": "f6e122cc-2e88-46ed-b7fb-7915a223c760"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "66/66 [==============================] - 4s 40ms/step - loss: 0.5408 - accuracy: 0.7465\n",
            "Epoch 2/75\n",
            "66/66 [==============================] - 2s 32ms/step - loss: 0.3379 - accuracy: 0.8755\n",
            "Epoch 3/75\n",
            "66/66 [==============================] - 3s 39ms/step - loss: 0.2247 - accuracy: 0.9189\n",
            "Epoch 4/75\n",
            "66/66 [==============================] - 2s 35ms/step - loss: 0.1325 - accuracy: 0.9496\n",
            "Epoch 5/75\n",
            "66/66 [==============================] - 2s 31ms/step - loss: 0.0705 - accuracy: 0.9779\n",
            "Epoch 6/75\n",
            "66/66 [==============================] - 2s 32ms/step - loss: 0.0512 - accuracy: 0.9815\n",
            "Epoch 7/75\n",
            "66/66 [==============================] - 2s 34ms/step - loss: 0.0277 - accuracy: 0.9914\n",
            "Epoch 8/75\n",
            "66/66 [==============================] - 2s 34ms/step - loss: 0.0165 - accuracy: 0.9974\n",
            "Epoch 9/75\n",
            "66/66 [==============================] - 3s 40ms/step - loss: 0.0107 - accuracy: 0.9983\n",
            "Epoch 10/75\n",
            "66/66 [==============================] - 2s 32ms/step - loss: 0.0084 - accuracy: 0.9988\n",
            "Epoch 11/75\n",
            "66/66 [==============================] - 2s 33ms/step - loss: 0.0173 - accuracy: 0.9954\n",
            "Epoch 12/75\n",
            "66/66 [==============================] - 2s 31ms/step - loss: 0.0032 - accuracy: 0.9998\n",
            "Epoch 13/75\n",
            "66/66 [==============================] - 2s 31ms/step - loss: 0.0017 - accuracy: 1.0000\n",
            "Epoch 14/75\n",
            "66/66 [==============================] - 3s 43ms/step - loss: 9.9657e-04 - accuracy: 1.0000\n",
            "Epoch 15/75\n",
            "66/66 [==============================] - 3s 43ms/step - loss: 8.2819e-04 - accuracy: 1.0000\n",
            "Epoch 16/75\n",
            "66/66 [==============================] - 2s 31ms/step - loss: 6.6089e-04 - accuracy: 1.0000\n",
            "Epoch 17/75\n",
            "66/66 [==============================] - 2s 32ms/step - loss: 5.3202e-04 - accuracy: 1.0000\n",
            "Epoch 18/75\n",
            "66/66 [==============================] - 2s 35ms/step - loss: 4.7340e-04 - accuracy: 1.0000\n",
            "Epoch 19/75\n",
            "66/66 [==============================] - 3s 44ms/step - loss: 3.9663e-04 - accuracy: 1.0000\n",
            "Epoch 20/75\n",
            "66/66 [==============================] - 2s 32ms/step - loss: 3.4761e-04 - accuracy: 1.0000\n",
            "Epoch 21/75\n",
            "66/66 [==============================] - 2s 32ms/step - loss: 3.0542e-04 - accuracy: 1.0000\n",
            "Epoch 22/75\n",
            "66/66 [==============================] - 2s 32ms/step - loss: 2.8892e-04 - accuracy: 1.0000\n",
            "Epoch 23/75\n",
            "66/66 [==============================] - 2s 32ms/step - loss: 2.4252e-04 - accuracy: 1.0000\n",
            "Epoch 24/75\n",
            "66/66 [==============================] - 3s 41ms/step - loss: 2.1807e-04 - accuracy: 1.0000\n",
            "Epoch 25/75\n",
            "66/66 [==============================] - 2s 33ms/step - loss: 1.9446e-04 - accuracy: 1.0000\n",
            "Epoch 26/75\n",
            "66/66 [==============================] - 2s 33ms/step - loss: 1.7788e-04 - accuracy: 1.0000\n",
            "Epoch 27/75\n",
            "66/66 [==============================] - 2s 33ms/step - loss: 1.6216e-04 - accuracy: 1.0000\n",
            "Epoch 28/75\n",
            "66/66 [==============================] - 2s 32ms/step - loss: 1.4780e-04 - accuracy: 1.0000\n",
            "Epoch 29/75\n",
            "66/66 [==============================] - 2s 34ms/step - loss: 1.3915e-04 - accuracy: 1.0000\n",
            "Epoch 30/75\n",
            "66/66 [==============================] - 3s 40ms/step - loss: 1.2359e-04 - accuracy: 1.0000\n",
            "Epoch 31/75\n",
            "66/66 [==============================] - 2s 33ms/step - loss: 1.1291e-04 - accuracy: 1.0000\n",
            "Epoch 32/75\n",
            "66/66 [==============================] - 2s 32ms/step - loss: 1.0526e-04 - accuracy: 1.0000\n",
            "Epoch 33/75\n",
            "66/66 [==============================] - 2s 32ms/step - loss: 9.6933e-05 - accuracy: 1.0000\n",
            "Epoch 34/75\n",
            "66/66 [==============================] - 2s 31ms/step - loss: 9.1528e-05 - accuracy: 1.0000\n",
            "Epoch 35/75\n",
            "66/66 [==============================] - 3s 43ms/step - loss: 8.3869e-05 - accuracy: 1.0000\n",
            "Epoch 36/75\n",
            "66/66 [==============================] - 2s 32ms/step - loss: 7.8481e-05 - accuracy: 1.0000\n",
            "Epoch 37/75\n",
            "66/66 [==============================] - 2s 32ms/step - loss: 7.2241e-05 - accuracy: 1.0000\n",
            "Epoch 38/75\n",
            "66/66 [==============================] - 2s 32ms/step - loss: 6.8253e-05 - accuracy: 1.0000\n",
            "Epoch 39/75\n",
            "66/66 [==============================] - 2s 30ms/step - loss: 6.3114e-05 - accuracy: 1.0000\n",
            "Epoch 40/75\n",
            "66/66 [==============================] - 3s 39ms/step - loss: 5.8903e-05 - accuracy: 1.0000\n",
            "Epoch 41/75\n",
            "66/66 [==============================] - 2s 35ms/step - loss: 5.5009e-05 - accuracy: 1.0000\n",
            "Epoch 42/75\n",
            "66/66 [==============================] - 2s 32ms/step - loss: 5.2325e-05 - accuracy: 1.0000\n",
            "Epoch 43/75\n",
            "66/66 [==============================] - 2s 32ms/step - loss: 4.8612e-05 - accuracy: 1.0000\n",
            "Epoch 44/75\n",
            "66/66 [==============================] - 2s 33ms/step - loss: 4.5960e-05 - accuracy: 1.0000\n",
            "Epoch 45/75\n",
            "66/66 [==============================] - 2s 33ms/step - loss: 4.3787e-05 - accuracy: 1.0000\n",
            "Epoch 46/75\n",
            "66/66 [==============================] - 3s 40ms/step - loss: 4.1375e-05 - accuracy: 1.0000\n",
            "Epoch 47/75\n",
            "66/66 [==============================] - 2s 32ms/step - loss: 3.8984e-05 - accuracy: 1.0000\n",
            "Epoch 48/75\n",
            "66/66 [==============================] - 2s 33ms/step - loss: 3.6507e-05 - accuracy: 1.0000\n",
            "Epoch 49/75\n",
            "66/66 [==============================] - 2s 32ms/step - loss: 3.4850e-05 - accuracy: 1.0000\n",
            "Epoch 50/75\n",
            "66/66 [==============================] - 2s 34ms/step - loss: 3.2171e-05 - accuracy: 1.0000\n",
            "Epoch 51/75\n",
            "66/66 [==============================] - 3s 42ms/step - loss: 3.1331e-05 - accuracy: 1.0000\n",
            "Epoch 52/75\n",
            "66/66 [==============================] - 2s 31ms/step - loss: 2.8898e-05 - accuracy: 1.0000\n",
            "Epoch 53/75\n",
            "66/66 [==============================] - 2s 31ms/step - loss: 2.7868e-05 - accuracy: 1.0000\n",
            "Epoch 54/75\n",
            "66/66 [==============================] - 2s 31ms/step - loss: 2.6152e-05 - accuracy: 1.0000\n",
            "Epoch 55/75\n",
            "66/66 [==============================] - 2s 31ms/step - loss: 2.4768e-05 - accuracy: 1.0000\n",
            "Epoch 56/75\n",
            "66/66 [==============================] - 3s 39ms/step - loss: 2.3455e-05 - accuracy: 1.0000\n",
            "Epoch 57/75\n",
            "66/66 [==============================] - 2s 36ms/step - loss: 2.2383e-05 - accuracy: 1.0000\n",
            "Epoch 58/75\n",
            "66/66 [==============================] - 2s 32ms/step - loss: 2.1141e-05 - accuracy: 1.0000\n",
            "Epoch 59/75\n",
            "66/66 [==============================] - 2s 33ms/step - loss: 2.0401e-05 - accuracy: 1.0000\n",
            "Epoch 60/75\n",
            "66/66 [==============================] - 2s 32ms/step - loss: 1.9454e-05 - accuracy: 1.0000\n",
            "Epoch 61/75\n",
            "66/66 [==============================] - 2s 32ms/step - loss: 1.8393e-05 - accuracy: 1.0000\n",
            "Epoch 62/75\n",
            "66/66 [==============================] - 3s 41ms/step - loss: 1.7492e-05 - accuracy: 1.0000\n",
            "Epoch 63/75\n",
            "66/66 [==============================] - 2s 32ms/step - loss: 1.6572e-05 - accuracy: 1.0000\n",
            "Epoch 64/75\n",
            "66/66 [==============================] - 2s 33ms/step - loss: 1.5863e-05 - accuracy: 1.0000\n",
            "Epoch 65/75\n",
            "66/66 [==============================] - 2s 31ms/step - loss: 1.4973e-05 - accuracy: 1.0000\n",
            "Epoch 66/75\n",
            "66/66 [==============================] - 2s 31ms/step - loss: 1.4393e-05 - accuracy: 1.0000\n",
            "Epoch 67/75\n",
            "66/66 [==============================] - 3s 43ms/step - loss: 1.3816e-05 - accuracy: 1.0000\n",
            "Epoch 68/75\n",
            "66/66 [==============================] - 2s 31ms/step - loss: 1.3125e-05 - accuracy: 1.0000\n",
            "Epoch 69/75\n",
            "66/66 [==============================] - 2s 32ms/step - loss: 1.2487e-05 - accuracy: 1.0000\n",
            "Epoch 70/75\n",
            "66/66 [==============================] - 2s 34ms/step - loss: 1.1821e-05 - accuracy: 1.0000\n",
            "Epoch 71/75\n",
            "66/66 [==============================] - 2s 34ms/step - loss: 1.1195e-05 - accuracy: 1.0000\n",
            "Epoch 72/75\n",
            "66/66 [==============================] - 3s 39ms/step - loss: 1.0906e-05 - accuracy: 1.0000\n",
            "Epoch 73/75\n",
            "66/66 [==============================] - 2s 37ms/step - loss: 1.0507e-05 - accuracy: 1.0000\n",
            "Epoch 74/75\n",
            "66/66 [==============================] - 2s 33ms/step - loss: 9.9209e-06 - accuracy: 1.0000\n",
            "Epoch 75/75\n",
            "66/66 [==============================] - 2s 32ms/step - loss: 9.6395e-06 - accuracy: 1.0000\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7967ae5d5990>"
            ]
          },
          "metadata": {},
          "execution_count": 62
        }
      ],
      "source": [
        "cnn.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "0OIHxWjXaVC1",
        "outputId": "ed3ec5a3-d508-4616-bbfb-5936f716b4df"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "33/33 [==============================] - 0s 7ms/step\n"
          ]
        }
      ],
      "source": [
        "pred = cnn.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "G-kV9E4naYQc",
        "outputId": "5ff8dcd3-b91e-4991-a9a6-9bf384232dfc"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.991362763915547,\n",
              " 1.0,\n",
              " 0.9838420107719928,\n",
              " 0.982663447567004,\n",
              " 0.98281115358556)"
            ]
          },
          "metadata": {},
          "execution_count": 64
        }
      ],
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "WNAdWrgTabkH",
        "outputId": "31bfbedd-139f-4f8f-da5b-de98d4890b2b"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9817813765182186"
            ]
          },
          "metadata": {},
          "execution_count": 65
        }
      ],
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "hZSrog7Yd9yh"
      },
      "source": [
        "**SMOTETomek**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "rVxd0Xa9eA99"
      },
      "outputs": [],
      "source": [
        "df1 = pd.read_csv('/content/LSA_TR.csv')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "V2UFbXaPeE-u"
      },
      "outputs": [],
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "38i7M4tTeKSU"
      },
      "outputs": [],
      "source": [
        "from imblearn.combine import SMOTETomek\n",
        "smt = SMOTETomek()\n",
        "X, Y = smt.fit_resample(X, Y)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "PDiedGGGegCl"
      },
      "outputs": [],
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "KWbjKZ_6euOV"
      },
      "outputs": [],
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "_GPStC6Xeype"
      },
      "outputs": [],
      "source": [
        "cnn = Sequential()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "0tBzTKvHe3Gg"
      },
      "outputs": [],
      "source": [
        "cnn.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))\n",
        "cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))\n",
        "cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "uygtVatHe70m"
      },
      "outputs": [],
      "source": [
        "cnn.add(MaxPool1D(pool_size=2))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "IyC536QXfApV"
      },
      "outputs": [],
      "source": [
        "cnn.add(Flatten())"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "7Kizq7DyfNWl"
      },
      "outputs": [],
      "source": [
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "m9dFNrEmfRgN"
      },
      "outputs": [],
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3tA4mka_fUqt",
        "outputId": "f32e7957-180c-4799-eaeb-ac18c1f96068"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "65/65 [==============================] - 5s 45ms/step - loss: 0.5942 - accuracy: 0.6861\n",
            "Epoch 2/75\n",
            "65/65 [==============================] - 3s 46ms/step - loss: 0.3454 - accuracy: 0.8601\n",
            "Epoch 3/75\n",
            "65/65 [==============================] - 3s 46ms/step - loss: 0.2263 - accuracy: 0.9213\n",
            "Epoch 4/75\n",
            "65/65 [==============================] - 4s 58ms/step - loss: 0.1466 - accuracy: 0.9516\n",
            "Epoch 5/75\n",
            "65/65 [==============================] - 2s 33ms/step - loss: 0.0922 - accuracy: 0.9722\n",
            "Epoch 6/75\n",
            "65/65 [==============================] - 2s 30ms/step - loss: 0.0829 - accuracy: 0.9739\n",
            "Epoch 7/75\n",
            "65/65 [==============================] - 2s 29ms/step - loss: 0.0540 - accuracy: 0.9848\n",
            "Epoch 8/75\n",
            "65/65 [==============================] - 2s 30ms/step - loss: 0.0359 - accuracy: 0.9906\n",
            "Epoch 9/75\n",
            "65/65 [==============================] - 2s 34ms/step - loss: 0.0200 - accuracy: 0.9961\n",
            "Epoch 10/75\n",
            "65/65 [==============================] - 2s 38ms/step - loss: 0.0213 - accuracy: 0.9949\n",
            "Epoch 11/75\n",
            "65/65 [==============================] - 2s 31ms/step - loss: 0.0098 - accuracy: 0.9978\n",
            "Epoch 12/75\n",
            "65/65 [==============================] - 2s 31ms/step - loss: 0.0059 - accuracy: 0.9993\n",
            "Epoch 13/75\n",
            "65/65 [==============================] - 2s 30ms/step - loss: 0.0038 - accuracy: 1.0000\n",
            "Epoch 14/75\n",
            "65/65 [==============================] - 2s 30ms/step - loss: 0.0024 - accuracy: 1.0000\n",
            "Epoch 15/75\n",
            "65/65 [==============================] - 3s 41ms/step - loss: 0.0018 - accuracy: 1.0000\n",
            "Epoch 16/75\n",
            "65/65 [==============================] - 2s 31ms/step - loss: 0.0016 - accuracy: 1.0000\n",
            "Epoch 17/75\n",
            "65/65 [==============================] - 2s 32ms/step - loss: 0.0015 - accuracy: 1.0000\n",
            "Epoch 18/75\n",
            "65/65 [==============================] - 2s 32ms/step - loss: 0.0010 - accuracy: 1.0000\n",
            "Epoch 19/75\n",
            "65/65 [==============================] - 2s 30ms/step - loss: 8.0971e-04 - accuracy: 1.0000\n",
            "Epoch 20/75\n",
            "65/65 [==============================] - 2s 30ms/step - loss: 7.4301e-04 - accuracy: 1.0000\n",
            "Epoch 21/75\n",
            "65/65 [==============================] - 3s 41ms/step - loss: 6.6568e-04 - accuracy: 1.0000\n",
            "Epoch 22/75\n",
            "65/65 [==============================] - 2s 30ms/step - loss: 5.6094e-04 - accuracy: 1.0000\n",
            "Epoch 23/75\n",
            "65/65 [==============================] - 2s 31ms/step - loss: 5.3350e-04 - accuracy: 1.0000\n",
            "Epoch 24/75\n",
            "65/65 [==============================] - 2s 31ms/step - loss: 4.3468e-04 - accuracy: 1.0000\n",
            "Epoch 25/75\n",
            "65/65 [==============================] - 2s 31ms/step - loss: 3.9348e-04 - accuracy: 1.0000\n",
            "Epoch 26/75\n",
            "65/65 [==============================] - 2s 37ms/step - loss: 3.5897e-04 - accuracy: 1.0000\n",
            "Epoch 27/75\n",
            "65/65 [==============================] - 3s 39ms/step - loss: 3.2514e-04 - accuracy: 1.0000\n",
            "Epoch 28/75\n",
            "65/65 [==============================] - 2s 32ms/step - loss: 2.9002e-04 - accuracy: 1.0000\n",
            "Epoch 29/75\n",
            "65/65 [==============================] - 2s 32ms/step - loss: 2.6378e-04 - accuracy: 1.0000\n",
            "Epoch 30/75\n",
            "65/65 [==============================] - 2s 32ms/step - loss: 2.3922e-04 - accuracy: 1.0000\n",
            "Epoch 31/75\n",
            "65/65 [==============================] - 2s 33ms/step - loss: 2.1679e-04 - accuracy: 1.0000\n",
            "Epoch 32/75\n",
            "65/65 [==============================] - 3s 42ms/step - loss: 1.9947e-04 - accuracy: 1.0000\n",
            "Epoch 33/75\n",
            "65/65 [==============================] - 2s 33ms/step - loss: 1.8325e-04 - accuracy: 1.0000\n",
            "Epoch 34/75\n",
            "65/65 [==============================] - 2s 32ms/step - loss: 1.7104e-04 - accuracy: 1.0000\n",
            "Epoch 35/75\n",
            "65/65 [==============================] - 2s 31ms/step - loss: 1.5989e-04 - accuracy: 1.0000\n",
            "Epoch 36/75\n",
            "65/65 [==============================] - 2s 31ms/step - loss: 1.4884e-04 - accuracy: 1.0000\n",
            "Epoch 37/75\n",
            "65/65 [==============================] - 3s 39ms/step - loss: 1.3751e-04 - accuracy: 1.0000\n",
            "Epoch 38/75\n",
            "65/65 [==============================] - 2s 36ms/step - loss: 1.3432e-04 - accuracy: 1.0000\n",
            "Epoch 39/75\n",
            "65/65 [==============================] - 2s 30ms/step - loss: 1.1866e-04 - accuracy: 1.0000\n",
            "Epoch 40/75\n",
            "65/65 [==============================] - 2s 31ms/step - loss: 1.1318e-04 - accuracy: 1.0000\n",
            "Epoch 41/75\n",
            "65/65 [==============================] - 2s 30ms/step - loss: 1.0362e-04 - accuracy: 1.0000\n",
            "Epoch 42/75\n",
            "65/65 [==============================] - 2s 31ms/step - loss: 9.8441e-05 - accuracy: 1.0000\n",
            "Epoch 43/75\n",
            "65/65 [==============================] - 3s 41ms/step - loss: 9.0987e-05 - accuracy: 1.0000\n",
            "Epoch 44/75\n",
            "65/65 [==============================] - 2s 31ms/step - loss: 8.3902e-05 - accuracy: 1.0000\n",
            "Epoch 45/75\n",
            "65/65 [==============================] - 2s 32ms/step - loss: 7.9482e-05 - accuracy: 1.0000\n",
            "Epoch 46/75\n",
            "65/65 [==============================] - 2s 30ms/step - loss: 7.4483e-05 - accuracy: 1.0000\n",
            "Epoch 47/75\n",
            "65/65 [==============================] - 2s 31ms/step - loss: 7.0687e-05 - accuracy: 1.0000\n",
            "Epoch 48/75\n",
            "65/65 [==============================] - 2s 30ms/step - loss: 6.7097e-05 - accuracy: 1.0000\n",
            "Epoch 49/75\n",
            "65/65 [==============================] - 3s 41ms/step - loss: 6.2871e-05 - accuracy: 1.0000\n",
            "Epoch 50/75\n",
            "65/65 [==============================] - 2s 30ms/step - loss: 5.9675e-05 - accuracy: 1.0000\n",
            "Epoch 51/75\n",
            "65/65 [==============================] - 2s 31ms/step - loss: 5.5360e-05 - accuracy: 1.0000\n",
            "Epoch 52/75\n",
            "65/65 [==============================] - 2s 31ms/step - loss: 5.2377e-05 - accuracy: 1.0000\n",
            "Epoch 53/75\n",
            "65/65 [==============================] - 2s 30ms/step - loss: 4.9936e-05 - accuracy: 1.0000\n",
            "Epoch 54/75\n",
            "65/65 [==============================] - 2s 37ms/step - loss: 4.6294e-05 - accuracy: 1.0000\n",
            "Epoch 55/75\n",
            "65/65 [==============================] - 2s 35ms/step - loss: 4.4381e-05 - accuracy: 1.0000\n",
            "Epoch 56/75\n",
            "65/65 [==============================] - 2s 30ms/step - loss: 4.2130e-05 - accuracy: 1.0000\n",
            "Epoch 57/75\n",
            "65/65 [==============================] - 2s 30ms/step - loss: 3.9542e-05 - accuracy: 1.0000\n",
            "Epoch 58/75\n",
            "65/65 [==============================] - 2s 30ms/step - loss: 3.8268e-05 - accuracy: 1.0000\n",
            "Epoch 59/75\n",
            "65/65 [==============================] - 2s 31ms/step - loss: 3.5516e-05 - accuracy: 1.0000\n",
            "Epoch 60/75\n",
            "65/65 [==============================] - 3s 41ms/step - loss: 3.4979e-05 - accuracy: 1.0000\n",
            "Epoch 61/75\n",
            "65/65 [==============================] - 2s 30ms/step - loss: 3.2108e-05 - accuracy: 1.0000\n",
            "Epoch 62/75\n",
            "65/65 [==============================] - 2s 31ms/step - loss: 3.0352e-05 - accuracy: 1.0000\n",
            "Epoch 63/75\n",
            "65/65 [==============================] - 2s 31ms/step - loss: 2.9088e-05 - accuracy: 1.0000\n",
            "Epoch 64/75\n",
            "65/65 [==============================] - 2s 31ms/step - loss: 2.7445e-05 - accuracy: 1.0000\n",
            "Epoch 65/75\n",
            "65/65 [==============================] - 2s 30ms/step - loss: 2.6837e-05 - accuracy: 1.0000\n",
            "Epoch 66/75\n",
            "65/65 [==============================] - 3s 41ms/step - loss: 2.4977e-05 - accuracy: 1.0000\n",
            "Epoch 67/75\n",
            "65/65 [==============================] - 2s 30ms/step - loss: 2.3968e-05 - accuracy: 1.0000\n",
            "Epoch 68/75\n",
            "65/65 [==============================] - 2s 30ms/step - loss: 2.2713e-05 - accuracy: 1.0000\n",
            "Epoch 69/75\n",
            "65/65 [==============================] - 2s 31ms/step - loss: 2.1451e-05 - accuracy: 1.0000\n",
            "Epoch 70/75\n",
            "65/65 [==============================] - 2s 31ms/step - loss: 2.0813e-05 - accuracy: 1.0000\n",
            "Epoch 71/75\n",
            "65/65 [==============================] - 2s 36ms/step - loss: 2.0268e-05 - accuracy: 1.0000\n",
            "Epoch 72/75\n",
            "65/65 [==============================] - 2s 37ms/step - loss: 1.8685e-05 - accuracy: 1.0000\n",
            "Epoch 73/75\n",
            "65/65 [==============================] - 2s 31ms/step - loss: 1.7973e-05 - accuracy: 1.0000\n",
            "Epoch 74/75\n",
            "65/65 [==============================] - 2s 30ms/step - loss: 1.7013e-05 - accuracy: 1.0000\n",
            "Epoch 75/75\n",
            "65/65 [==============================] - 2s 30ms/step - loss: 1.6283e-05 - accuracy: 1.0000\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7967ad25fa60>"
            ]
          },
          "metadata": {},
          "execution_count": 77
        }
      ],
      "source": [
        "cnn.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "W-LuZYYwfYUt",
        "outputId": "2be9a702-75d5-4f12-d22c-dcc9512ef26e"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "33/33 [==============================] - 0s 6ms/step\n"
          ]
        }
      ],
      "source": [
        "pred = cnn.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "VdSUOjY-fbEF",
        "outputId": "8f603b2e-92db-4de2-e157-fc0b77bfa89e"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9912790697674418,\n",
              " 1.0,\n",
              " 0.9825581395348837,\n",
              " 0.9825581395348837,\n",
              " 0.9827076298239908)"
            ]
          },
          "metadata": {},
          "execution_count": 79
        }
      ],
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "B2SDgmi-fd9F",
        "outputId": "23eb93bd-6811-4140-82d0-47376b5cbc08"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9828571428571429"
            ]
          },
          "metadata": {},
          "execution_count": 80
        }
      ],
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**NearMiss**"
      ],
      "metadata": {
        "id": "pdx9Dp8O1usl"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/LSA_TR.csv')"
      ],
      "metadata": {
        "id": "OHKIV6GI1xfL"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "4nbVUTS816QW"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.under_sampling import NearMiss\n",
        "nm = NearMiss()\n",
        "X, Y = nm.fit_resample(X, Y)"
      ],
      "metadata": {
        "id": "N-VV3rqe19Xr"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ],
      "metadata": {
        "id": "iDpX9fqz2COd"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ],
      "metadata": {
        "id": "7Qphh-7F2FLF"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn = Sequential()"
      ],
      "metadata": {
        "id": "S7VbW__c2IBm"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))\n",
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu'))\n",
        "cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))"
      ],
      "metadata": {
        "id": "AsHR1TmS2Kvc"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "y5Fj4XUM2N4V"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Flatten())"
      ],
      "metadata": {
        "id": "OOK7k8PX2Q-N"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "NeNaYV8D2Uh7"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "Vt9JhjtK2XbD"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "mS_JHM9g2aZi",
        "outputId": "4a5d1dfb-253c-47a9-8f65-6bed6e98f93b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "17/17 [==============================] - 7s 224ms/step - loss: 0.6570 - accuracy: 0.5262\n",
            "Epoch 2/75\n",
            "17/17 [==============================] - 4s 260ms/step - loss: 0.4607 - accuracy: 0.8198\n",
            "Epoch 3/75\n",
            "17/17 [==============================] - 4s 216ms/step - loss: 0.2957 - accuracy: 0.8973\n",
            "Epoch 4/75\n",
            "17/17 [==============================] - 4s 245ms/step - loss: 0.2823 - accuracy: 0.8924\n",
            "Epoch 5/75\n",
            "17/17 [==============================] - 6s 358ms/step - loss: 0.2410 - accuracy: 0.9050\n",
            "Epoch 6/75\n",
            "17/17 [==============================] - 4s 234ms/step - loss: 0.2325 - accuracy: 0.9060\n",
            "Epoch 7/75\n",
            "17/17 [==============================] - 3s 199ms/step - loss: 0.3200 - accuracy: 0.8421\n",
            "Epoch 8/75\n",
            "17/17 [==============================] - 3s 164ms/step - loss: 0.2267 - accuracy: 0.8915\n",
            "Epoch 9/75\n",
            "17/17 [==============================] - 4s 258ms/step - loss: 0.1498 - accuracy: 0.9138\n",
            "Epoch 10/75\n",
            "17/17 [==============================] - 4s 220ms/step - loss: 0.1140 - accuracy: 0.9283\n",
            "Epoch 11/75\n",
            "17/17 [==============================] - 2s 146ms/step - loss: 0.1006 - accuracy: 0.9322\n",
            "Epoch 12/75\n",
            "17/17 [==============================] - 3s 188ms/step - loss: 0.0840 - accuracy: 0.9419\n",
            "Epoch 13/75\n",
            "17/17 [==============================] - 3s 167ms/step - loss: 0.0873 - accuracy: 0.9748\n",
            "Epoch 14/75\n",
            "17/17 [==============================] - 2s 147ms/step - loss: 0.1008 - accuracy: 0.9671\n",
            "Epoch 15/75\n",
            "17/17 [==============================] - 2s 146ms/step - loss: 0.0728 - accuracy: 0.9816\n",
            "Epoch 16/75\n",
            "17/17 [==============================] - 3s 177ms/step - loss: 0.0523 - accuracy: 0.9874\n",
            "Epoch 17/75\n",
            "17/17 [==============================] - 4s 265ms/step - loss: 0.0339 - accuracy: 0.9932\n",
            "Epoch 18/75\n",
            "17/17 [==============================] - 4s 234ms/step - loss: 0.0269 - accuracy: 0.9961\n",
            "Epoch 19/75\n",
            "17/17 [==============================] - 3s 189ms/step - loss: 0.0218 - accuracy: 0.9981\n",
            "Epoch 20/75\n",
            "17/17 [==============================] - 5s 271ms/step - loss: 0.0186 - accuracy: 0.9981\n",
            "Epoch 21/75\n",
            "17/17 [==============================] - 3s 190ms/step - loss: 0.0143 - accuracy: 0.9990\n",
            "Epoch 22/75\n",
            "17/17 [==============================] - 3s 158ms/step - loss: 0.0128 - accuracy: 0.9990\n",
            "Epoch 23/75\n",
            "17/17 [==============================] - 3s 180ms/step - loss: 0.0227 - accuracy: 0.9942\n",
            "Epoch 24/75\n",
            "17/17 [==============================] - 5s 272ms/step - loss: 0.0123 - accuracy: 0.9990\n",
            "Epoch 25/75\n",
            "17/17 [==============================] - 4s 215ms/step - loss: 0.0111 - accuracy: 0.9990\n",
            "Epoch 26/75\n",
            "17/17 [==============================] - 3s 164ms/step - loss: 0.0088 - accuracy: 0.9990\n",
            "Epoch 27/75\n",
            "17/17 [==============================] - 4s 216ms/step - loss: 0.0085 - accuracy: 0.9990\n",
            "Epoch 28/75\n",
            "17/17 [==============================] - 4s 245ms/step - loss: 0.0082 - accuracy: 0.9990\n",
            "Epoch 29/75\n",
            "17/17 [==============================] - 3s 162ms/step - loss: 0.0073 - accuracy: 0.9990\n",
            "Epoch 30/75\n",
            "17/17 [==============================] - 4s 207ms/step - loss: 0.0066 - accuracy: 0.9990\n",
            "Epoch 31/75\n",
            "17/17 [==============================] - 6s 337ms/step - loss: 0.0065 - accuracy: 0.9990\n",
            "Epoch 32/75\n",
            "17/17 [==============================] - 5s 290ms/step - loss: 0.0061 - accuracy: 0.9990\n",
            "Epoch 33/75\n",
            "17/17 [==============================] - 3s 160ms/step - loss: 0.0059 - accuracy: 0.9990\n",
            "Epoch 34/75\n",
            "17/17 [==============================] - 3s 192ms/step - loss: 0.0058 - accuracy: 0.9990\n",
            "Epoch 35/75\n",
            "17/17 [==============================] - 3s 162ms/step - loss: 0.0056 - accuracy: 0.9990\n",
            "Epoch 36/75\n",
            "17/17 [==============================] - 2s 144ms/step - loss: 0.0055 - accuracy: 0.9990\n",
            "Epoch 37/75\n",
            "17/17 [==============================] - 2s 145ms/step - loss: 0.0053 - accuracy: 0.9990\n",
            "Epoch 38/75\n",
            "17/17 [==============================] - 2s 145ms/step - loss: 0.0053 - accuracy: 0.9990\n",
            "Epoch 39/75\n",
            "17/17 [==============================] - 4s 215ms/step - loss: 0.0051 - accuracy: 0.9990\n",
            "Epoch 40/75\n",
            "17/17 [==============================] - 2s 145ms/step - loss: 0.0050 - accuracy: 0.9990\n",
            "Epoch 41/75\n",
            "17/17 [==============================] - 2s 144ms/step - loss: 0.0049 - accuracy: 0.9990\n",
            "Epoch 42/75\n",
            "17/17 [==============================] - 2s 145ms/step - loss: 0.0048 - accuracy: 0.9990\n",
            "Epoch 43/75\n",
            "17/17 [==============================] - 2s 145ms/step - loss: 0.0047 - accuracy: 0.9990\n",
            "Epoch 44/75\n",
            "17/17 [==============================] - 4s 209ms/step - loss: 0.0046 - accuracy: 0.9990\n",
            "Epoch 45/75\n",
            "17/17 [==============================] - 2s 145ms/step - loss: 0.0045 - accuracy: 0.9990\n",
            "Epoch 46/75\n",
            "17/17 [==============================] - 2s 146ms/step - loss: 0.0044 - accuracy: 0.9990\n",
            "Epoch 47/75\n",
            "17/17 [==============================] - 2s 145ms/step - loss: 0.0044 - accuracy: 0.9990\n",
            "Epoch 48/75\n",
            "17/17 [==============================] - 3s 170ms/step - loss: 0.0042 - accuracy: 0.9990\n",
            "Epoch 49/75\n",
            "17/17 [==============================] - 4s 215ms/step - loss: 0.0041 - accuracy: 0.9990\n",
            "Epoch 50/75\n",
            "17/17 [==============================] - 2s 145ms/step - loss: 0.0040 - accuracy: 0.9990\n",
            "Epoch 51/75\n",
            "17/17 [==============================] - 2s 144ms/step - loss: 0.0039 - accuracy: 0.9990\n",
            "Epoch 52/75\n",
            "17/17 [==============================] - 2s 144ms/step - loss: 0.0039 - accuracy: 0.9990\n",
            "Epoch 53/75\n",
            "17/17 [==============================] - 4s 210ms/step - loss: 0.0037 - accuracy: 0.9990\n",
            "Epoch 54/75\n",
            "17/17 [==============================] - 3s 144ms/step - loss: 0.0037 - accuracy: 0.9990\n",
            "Epoch 55/75\n",
            "17/17 [==============================] - 2s 144ms/step - loss: 0.0035 - accuracy: 0.9990\n",
            "Epoch 56/75\n",
            "17/17 [==============================] - 2s 145ms/step - loss: 0.0034 - accuracy: 0.9990\n",
            "Epoch 57/75\n",
            "17/17 [==============================] - 2s 144ms/step - loss: 0.0032 - accuracy: 0.9990\n",
            "Epoch 58/75\n",
            "17/17 [==============================] - 4s 216ms/step - loss: 0.0031 - accuracy: 0.9990\n",
            "Epoch 59/75\n",
            "17/17 [==============================] - 2s 144ms/step - loss: 0.0030 - accuracy: 0.9990\n",
            "Epoch 60/75\n",
            "17/17 [==============================] - 2s 145ms/step - loss: 0.0029 - accuracy: 0.9990\n",
            "Epoch 61/75\n",
            "17/17 [==============================] - 2s 144ms/step - loss: 0.0026 - accuracy: 0.9990\n",
            "Epoch 62/75\n",
            "17/17 [==============================] - 3s 160ms/step - loss: 0.0025 - accuracy: 0.9990\n",
            "Epoch 63/75\n",
            "17/17 [==============================] - 3s 193ms/step - loss: 0.0023 - accuracy: 0.9990\n",
            "Epoch 64/75\n",
            "17/17 [==============================] - 3s 159ms/step - loss: 0.0022 - accuracy: 0.9990\n",
            "Epoch 65/75\n",
            "17/17 [==============================] - 3s 171ms/step - loss: 0.0021 - accuracy: 0.9990\n",
            "Epoch 66/75\n",
            "17/17 [==============================] - 3s 191ms/step - loss: 0.0019 - accuracy: 0.9990\n",
            "Epoch 67/75\n",
            "17/17 [==============================] - 4s 216ms/step - loss: 0.0017 - accuracy: 0.9990\n",
            "Epoch 68/75\n",
            "17/17 [==============================] - 2s 144ms/step - loss: 0.0015 - accuracy: 0.9990\n",
            "Epoch 69/75\n",
            "17/17 [==============================] - 2s 147ms/step - loss: 0.0015 - accuracy: 0.9990\n",
            "Epoch 70/75\n",
            "17/17 [==============================] - 2s 144ms/step - loss: 0.0013 - accuracy: 0.9990\n",
            "Epoch 71/75\n",
            "17/17 [==============================] - 4s 222ms/step - loss: 0.0012 - accuracy: 1.0000\n",
            "Epoch 72/75\n",
            "17/17 [==============================] - 3s 169ms/step - loss: 0.0011 - accuracy: 0.9990\n",
            "Epoch 73/75\n",
            "17/17 [==============================] - 2s 145ms/step - loss: 0.0011 - accuracy: 0.9990\n",
            "Epoch 74/75\n",
            "17/17 [==============================] - 2s 145ms/step - loss: 0.0010 - accuracy: 0.9990\n",
            "Epoch 75/75\n",
            "17/17 [==============================] - 2s 144ms/step - loss: 0.0010 - accuracy: 1.0000\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x794cdc4dada0>"
            ]
          },
          "metadata": {},
          "execution_count": 14
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cnn.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "qboBtFhw2fhO",
        "outputId": "2e5f7cbd-d99c-4cca-e7de-c1019cea9049"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "9/9 [==============================] - 0s 19ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "L7PZ7nIE2iHT",
        "outputId": "9968159f-33ce-42a9-bd7b-de224ce24d48"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9806201550387597,\n",
              " 0.9924242424242424,\n",
              " 0.9703703703703703,\n",
              " 0.9611983396498827,\n",
              " 0.9614589326361405)"
            ]
          },
          "metadata": {},
          "execution_count": 16
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-y-5K9aI2lCb",
        "outputId": "def382bd-2c48-449a-f127-a14cdcc8f8a9"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9682539682539683"
            ]
          },
          "metadata": {},
          "execution_count": 17
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **CNN(NMBroto)**"
      ],
      "metadata": {
        "id": "vOo7m3UdDDdE"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Imbalanced**"
      ],
      "metadata": {
        "id": "VcQviomHCPM9"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/NMB-TR.csv')"
      ],
      "metadata": {
        "id": "k9CRblklDLzz"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "fxGrRINfDV6T"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ],
      "metadata": {
        "id": "fyPqSzpJDYfb"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ],
      "metadata": {
        "id": "8yiXh9xkDa9u"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn = Sequential()"
      ],
      "metadata": {
        "id": "__cWqdRvDdBz"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))\n",
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu'))\n",
        "cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))"
      ],
      "metadata": {
        "id": "sapE3489De3j"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "ybV7cGfdDhDT"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Flatten())"
      ],
      "metadata": {
        "id": "l0iXda6wDjRD"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "l7f9uX2DDlTc"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "blxP7Gj3DnD8"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "LFMkvKwKDpOv",
        "outputId": "36a5812d-dcdc-4524-8e3f-01df1996b87c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "41/41 [==============================] - 5s 81ms/step - loss: 0.5507 - accuracy: 0.7796\n",
            "Epoch 2/75\n",
            "41/41 [==============================] - 3s 80ms/step - loss: 0.5092 - accuracy: 0.7990\n",
            "Epoch 3/75\n",
            "41/41 [==============================] - 3s 72ms/step - loss: 0.4786 - accuracy: 0.7990\n",
            "Epoch 4/75\n",
            "41/41 [==============================] - 4s 96ms/step - loss: 0.4434 - accuracy: 0.7998\n",
            "Epoch 5/75\n",
            "41/41 [==============================] - 3s 84ms/step - loss: 0.4120 - accuracy: 0.8176\n",
            "Epoch 6/75\n",
            "41/41 [==============================] - 2s 60ms/step - loss: 0.3969 - accuracy: 0.8311\n",
            "Epoch 7/75\n",
            "41/41 [==============================] - 3s 66ms/step - loss: 0.3571 - accuracy: 0.8493\n",
            "Epoch 8/75\n",
            "41/41 [==============================] - 4s 95ms/step - loss: 0.3267 - accuracy: 0.8703\n",
            "Epoch 9/75\n",
            "41/41 [==============================] - 3s 80ms/step - loss: 0.2974 - accuracy: 0.8826\n",
            "Epoch 10/75\n",
            "41/41 [==============================] - 4s 87ms/step - loss: 0.2658 - accuracy: 0.8908\n",
            "Epoch 11/75\n",
            "41/41 [==============================] - 5s 125ms/step - loss: 0.2320 - accuracy: 0.9129\n",
            "Epoch 12/75\n",
            "41/41 [==============================] - 4s 87ms/step - loss: 0.2052 - accuracy: 0.9272\n",
            "Epoch 13/75\n",
            "41/41 [==============================] - 3s 68ms/step - loss: 0.2024 - accuracy: 0.9276\n",
            "Epoch 14/75\n",
            "41/41 [==============================] - 2s 55ms/step - loss: 0.1531 - accuracy: 0.9411\n",
            "Epoch 15/75\n",
            "41/41 [==============================] - 3s 83ms/step - loss: 0.1312 - accuracy: 0.9535\n",
            "Epoch 16/75\n",
            "41/41 [==============================] - 2s 55ms/step - loss: 0.1241 - accuracy: 0.9547\n",
            "Epoch 17/75\n",
            "41/41 [==============================] - 2s 55ms/step - loss: 0.1222 - accuracy: 0.9516\n",
            "Epoch 18/75\n",
            "41/41 [==============================] - 2s 55ms/step - loss: 0.0893 - accuracy: 0.9694\n",
            "Epoch 19/75\n",
            "41/41 [==============================] - 2s 55ms/step - loss: 0.0710 - accuracy: 0.9756\n",
            "Epoch 20/75\n",
            "41/41 [==============================] - 3s 76ms/step - loss: 0.0677 - accuracy: 0.9764\n",
            "Epoch 21/75\n",
            "41/41 [==============================] - 3s 60ms/step - loss: 0.0474 - accuracy: 0.9868\n",
            "Epoch 22/75\n",
            "41/41 [==============================] - 2s 55ms/step - loss: 0.0532 - accuracy: 0.9833\n",
            "Epoch 23/75\n",
            "41/41 [==============================] - 2s 56ms/step - loss: 0.0428 - accuracy: 0.9892\n",
            "Epoch 24/75\n",
            "41/41 [==============================] - 2s 55ms/step - loss: 0.0324 - accuracy: 0.9899\n",
            "Epoch 25/75\n",
            "41/41 [==============================] - 3s 68ms/step - loss: 0.0324 - accuracy: 0.9915\n",
            "Epoch 26/75\n",
            "41/41 [==============================] - 3s 68ms/step - loss: 0.0344 - accuracy: 0.9907\n",
            "Epoch 27/75\n",
            "41/41 [==============================] - 2s 55ms/step - loss: 0.0322 - accuracy: 0.9923\n",
            "Epoch 28/75\n",
            "41/41 [==============================] - 2s 55ms/step - loss: 0.0282 - accuracy: 0.9919\n",
            "Epoch 29/75\n",
            "41/41 [==============================] - 2s 56ms/step - loss: 0.0194 - accuracy: 0.9950\n",
            "Epoch 30/75\n",
            "41/41 [==============================] - 3s 62ms/step - loss: 0.0148 - accuracy: 0.9969\n",
            "Epoch 31/75\n",
            "41/41 [==============================] - 4s 100ms/step - loss: 0.0178 - accuracy: 0.9938\n",
            "Epoch 32/75\n",
            "41/41 [==============================] - 4s 90ms/step - loss: 0.0209 - accuracy: 0.9938\n",
            "Epoch 33/75\n",
            "41/41 [==============================] - 4s 85ms/step - loss: 0.0252 - accuracy: 0.9926\n",
            "Epoch 34/75\n",
            "41/41 [==============================] - 6s 138ms/step - loss: 0.0465 - accuracy: 0.9868\n",
            "Epoch 35/75\n",
            "41/41 [==============================] - 4s 94ms/step - loss: 0.0543 - accuracy: 0.9830\n",
            "Epoch 36/75\n",
            "41/41 [==============================] - 5s 111ms/step - loss: 0.0238 - accuracy: 0.9934\n",
            "Epoch 37/75\n",
            "41/41 [==============================] - 5s 115ms/step - loss: 0.0107 - accuracy: 0.9988\n",
            "Epoch 38/75\n",
            "41/41 [==============================] - 4s 87ms/step - loss: 0.0070 - accuracy: 0.9981\n",
            "Epoch 39/75\n",
            "41/41 [==============================] - 3s 81ms/step - loss: 0.0077 - accuracy: 0.9973\n",
            "Epoch 40/75\n",
            "41/41 [==============================] - 4s 107ms/step - loss: 0.0147 - accuracy: 0.9965\n",
            "Epoch 41/75\n",
            "41/41 [==============================] - 4s 88ms/step - loss: 0.0162 - accuracy: 0.9946\n",
            "Epoch 42/75\n",
            "41/41 [==============================] - 3s 74ms/step - loss: 0.0160 - accuracy: 0.9957\n",
            "Epoch 43/75\n",
            "41/41 [==============================] - 2s 61ms/step - loss: 0.0143 - accuracy: 0.9961\n",
            "Epoch 44/75\n",
            "41/41 [==============================] - 4s 110ms/step - loss: 0.0221 - accuracy: 0.9934\n",
            "Epoch 45/75\n",
            "41/41 [==============================] - 3s 81ms/step - loss: 0.0205 - accuracy: 0.9930\n",
            "Epoch 46/75\n",
            "41/41 [==============================] - 3s 69ms/step - loss: 0.0196 - accuracy: 0.9938\n",
            "Epoch 47/75\n",
            "41/41 [==============================] - 3s 81ms/step - loss: 0.0432 - accuracy: 0.9849\n",
            "Epoch 48/75\n",
            "41/41 [==============================] - 5s 110ms/step - loss: 0.0470 - accuracy: 0.9818\n",
            "Epoch 49/75\n",
            "41/41 [==============================] - 3s 81ms/step - loss: 0.0225 - accuracy: 0.9923\n",
            "Epoch 50/75\n",
            "41/41 [==============================] - 4s 87ms/step - loss: 0.0317 - accuracy: 0.9907\n",
            "Epoch 51/75\n",
            "41/41 [==============================] - 4s 101ms/step - loss: 0.0339 - accuracy: 0.9892\n",
            "Epoch 52/75\n",
            "41/41 [==============================] - 4s 93ms/step - loss: 0.0230 - accuracy: 0.9930\n",
            "Epoch 53/75\n",
            "41/41 [==============================] - 4s 97ms/step - loss: 0.0122 - accuracy: 0.9957\n",
            "Epoch 54/75\n",
            "41/41 [==============================] - 4s 100ms/step - loss: 0.0133 - accuracy: 0.9973\n",
            "Epoch 55/75\n",
            "41/41 [==============================] - 4s 97ms/step - loss: 0.0056 - accuracy: 0.9973\n",
            "Epoch 56/75\n",
            "41/41 [==============================] - 4s 89ms/step - loss: 0.0067 - accuracy: 0.9977\n",
            "Epoch 57/75\n",
            "41/41 [==============================] - 3s 83ms/step - loss: 0.0067 - accuracy: 0.9977\n",
            "Epoch 58/75\n",
            "41/41 [==============================] - 5s 134ms/step - loss: 0.0066 - accuracy: 0.9973\n",
            "Epoch 59/75\n",
            "41/41 [==============================] - 4s 95ms/step - loss: 0.0074 - accuracy: 0.9977\n",
            "Epoch 60/75\n",
            "41/41 [==============================] - 4s 100ms/step - loss: 0.0054 - accuracy: 0.9985\n",
            "Epoch 61/75\n",
            "41/41 [==============================] - 4s 108ms/step - loss: 0.0070 - accuracy: 0.9977\n",
            "Epoch 62/75\n",
            "41/41 [==============================] - 4s 87ms/step - loss: 0.0049 - accuracy: 0.9985\n",
            "Epoch 63/75\n",
            "41/41 [==============================] - 4s 85ms/step - loss: 0.0059 - accuracy: 0.9973\n",
            "Epoch 64/75\n",
            "41/41 [==============================] - 3s 66ms/step - loss: 0.0088 - accuracy: 0.9977\n",
            "Epoch 65/75\n",
            "41/41 [==============================] - 5s 116ms/step - loss: 0.0053 - accuracy: 0.9988\n",
            "Epoch 66/75\n",
            "41/41 [==============================] - 3s 85ms/step - loss: 0.0097 - accuracy: 0.9961\n",
            "Epoch 67/75\n",
            "41/41 [==============================] - 2s 60ms/step - loss: 0.0060 - accuracy: 0.9981\n",
            "Epoch 68/75\n",
            "41/41 [==============================] - 4s 97ms/step - loss: 0.0082 - accuracy: 0.9973\n",
            "Epoch 69/75\n",
            "41/41 [==============================] - 4s 93ms/step - loss: 0.0148 - accuracy: 0.9961\n",
            "Epoch 70/75\n",
            "41/41 [==============================] - 3s 86ms/step - loss: 0.0105 - accuracy: 0.9965\n",
            "Epoch 71/75\n",
            "41/41 [==============================] - 3s 82ms/step - loss: 0.0066 - accuracy: 0.9981\n",
            "Epoch 72/75\n",
            "41/41 [==============================] - 4s 102ms/step - loss: 0.0070 - accuracy: 0.9981\n",
            "Epoch 73/75\n",
            "41/41 [==============================] - 3s 77ms/step - loss: 0.0056 - accuracy: 0.9985\n",
            "Epoch 74/75\n",
            "41/41 [==============================] - 3s 79ms/step - loss: 0.0073 - accuracy: 0.9985\n",
            "Epoch 75/75\n",
            "41/41 [==============================] - 3s 79ms/step - loss: 0.0085 - accuracy: 0.9977\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x794cccdba740>"
            ]
          },
          "metadata": {},
          "execution_count": 37
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cnn.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "c_rHP0QFDroF",
        "outputId": "5aa75935-16f9-4119-98bb-d2f735f63414"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "21/21 [==============================] - 0s 6ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "h-iApOXcDudb",
        "outputId": "7b5fe832-149e-4ef1-a320-a42be79eb7a4"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9550387596899225,\n",
              " 0.9523809523809523,\n",
              " 0.8391608391608392,\n",
              " 0.8639329594308535,\n",
              " 0.8666944028106094)"
            ]
          },
          "metadata": {},
          "execution_count": 39
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8SuWzqJ_Dydb",
        "outputId": "bef20c8e-c689-4fcd-89c9-7ef0f83df1a6"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9556840077071291"
            ]
          },
          "metadata": {},
          "execution_count": 40
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Test**"
      ],
      "metadata": {
        "id": "KSGmYSjgCV-D"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/NMB-TR.csv')\n",
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "wD5r14a4CXqU"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.3, random_state = 1)"
      ],
      "metadata": {
        "id": "yoG0gI8SnbVZ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "sample_size = xtrain.shape[0] # number of samples in train set\n",
        "time_steps  = xtrain.shape[1] # number of features in train set\n",
        "input_dimension = 1               # each feature is represented by 1 number"
      ],
      "metadata": {
        "id": "dJC-SKxhCiK_"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_data_reshaped = xtrain.values.reshape(sample_size,time_steps,input_dimension)\n",
        "n_timesteps = train_data_reshaped.shape[1]\n",
        "n_features  = train_data_reshaped.shape[2]"
      ],
      "metadata": {
        "id": "oJA50PdYCkeq"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn = Sequential()"
      ],
      "metadata": {
        "id": "cN3HpBRzCmqk"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))\n",
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu'))\n",
        "cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))"
      ],
      "metadata": {
        "id": "82H9x-IWCo0B"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "eMu9807aCrLJ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Flatten())"
      ],
      "metadata": {
        "id": "_X16QWAXCtXD"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "uNzDV9vlCvmF"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "Tglb6FS_CxxK"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.fit(xtrain, ytrain, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "tOEFq8Y1C0FI",
        "outputId": "56294f2a-674d-4183-d30b-2e0c4df69442"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "36/36 [==============================] - 8s 127ms/step - loss: 0.5460 - accuracy: 0.8007\n",
            "Epoch 2/75\n",
            "36/36 [==============================] - 3s 94ms/step - loss: 0.4880 - accuracy: 0.8127\n",
            "Epoch 3/75\n",
            "36/36 [==============================] - 4s 101ms/step - loss: 0.4691 - accuracy: 0.8127\n",
            "Epoch 4/75\n",
            "36/36 [==============================] - 4s 125ms/step - loss: 0.4545 - accuracy: 0.8127\n",
            "Epoch 5/75\n",
            "36/36 [==============================] - 3s 95ms/step - loss: 0.4227 - accuracy: 0.8127\n",
            "Epoch 6/75\n",
            "36/36 [==============================] - 4s 116ms/step - loss: 0.3969 - accuracy: 0.8127\n",
            "Epoch 7/75\n",
            "36/36 [==============================] - 5s 154ms/step - loss: 0.3850 - accuracy: 0.8127\n",
            "Epoch 8/75\n",
            "36/36 [==============================] - 4s 99ms/step - loss: 0.3557 - accuracy: 0.8127\n",
            "Epoch 9/75\n",
            "36/36 [==============================] - 3s 92ms/step - loss: 0.3470 - accuracy: 0.8127\n",
            "Epoch 10/75\n",
            "36/36 [==============================] - 4s 99ms/step - loss: 0.3282 - accuracy: 0.8339\n",
            "Epoch 11/75\n",
            "36/36 [==============================] - 4s 121ms/step - loss: 0.3208 - accuracy: 0.8432\n",
            "Epoch 12/75\n",
            "36/36 [==============================] - 3s 93ms/step - loss: 0.2897 - accuracy: 0.8676\n",
            "Epoch 13/75\n",
            "36/36 [==============================] - 3s 91ms/step - loss: 0.2708 - accuracy: 0.8702\n",
            "Epoch 14/75\n",
            "36/36 [==============================] - 4s 115ms/step - loss: 0.2539 - accuracy: 0.8884\n",
            "Epoch 15/75\n",
            "36/36 [==============================] - 4s 98ms/step - loss: 0.2357 - accuracy: 0.8986\n",
            "Epoch 16/75\n",
            "36/36 [==============================] - 3s 92ms/step - loss: 0.2192 - accuracy: 0.9145\n",
            "Epoch 17/75\n",
            "36/36 [==============================] - 3s 96ms/step - loss: 0.2029 - accuracy: 0.9141\n",
            "Epoch 18/75\n",
            "36/36 [==============================] - 4s 126ms/step - loss: 0.1880 - accuracy: 0.9367\n",
            "Epoch 19/75\n",
            "36/36 [==============================] - 3s 96ms/step - loss: 0.1777 - accuracy: 0.9469\n",
            "Epoch 20/75\n",
            "36/36 [==============================] - 4s 99ms/step - loss: 0.1717 - accuracy: 0.9513\n",
            "Epoch 21/75\n",
            "36/36 [==============================] - 4s 103ms/step - loss: 0.1595 - accuracy: 0.9557\n",
            "Epoch 22/75\n",
            "36/36 [==============================] - 4s 115ms/step - loss: 0.1497 - accuracy: 0.9659\n",
            "Epoch 23/75\n",
            "36/36 [==============================] - 2s 62ms/step - loss: 0.1407 - accuracy: 0.9761\n",
            "Epoch 24/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.1320 - accuracy: 0.9779\n",
            "Epoch 25/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.1247 - accuracy: 0.9823\n",
            "Epoch 26/75\n",
            "36/36 [==============================] - 3s 81ms/step - loss: 0.1162 - accuracy: 0.9876\n",
            "Epoch 27/75\n",
            "36/36 [==============================] - 3s 79ms/step - loss: 0.1119 - accuracy: 0.9863\n",
            "Epoch 28/75\n",
            "36/36 [==============================] - 2s 62ms/step - loss: 0.1069 - accuracy: 0.9894\n",
            "Epoch 29/75\n",
            "36/36 [==============================] - 3s 76ms/step - loss: 0.1040 - accuracy: 0.9916\n",
            "Epoch 30/75\n",
            "36/36 [==============================] - 2s 62ms/step - loss: 0.1009 - accuracy: 0.9907\n",
            "Epoch 31/75\n",
            "36/36 [==============================] - 3s 70ms/step - loss: 0.0936 - accuracy: 0.9942\n",
            "Epoch 32/75\n",
            "36/36 [==============================] - 3s 87ms/step - loss: 0.0889 - accuracy: 0.9934\n",
            "Epoch 33/75\n",
            "36/36 [==============================] - 2s 63ms/step - loss: 0.0851 - accuracy: 0.9965\n",
            "Epoch 34/75\n",
            "36/36 [==============================] - 2s 62ms/step - loss: 0.0831 - accuracy: 0.9965\n",
            "Epoch 35/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.0814 - accuracy: 0.9960\n",
            "Epoch 36/75\n",
            "36/36 [==============================] - 2s 62ms/step - loss: 0.0779 - accuracy: 0.9973\n",
            "Epoch 37/75\n",
            "36/36 [==============================] - 3s 96ms/step - loss: 0.0787 - accuracy: 0.9956\n",
            "Epoch 38/75\n",
            "36/36 [==============================] - 2s 62ms/step - loss: 0.0766 - accuracy: 0.9965\n",
            "Epoch 39/75\n",
            "36/36 [==============================] - 2s 62ms/step - loss: 0.0727 - accuracy: 0.9969\n",
            "Epoch 40/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.0709 - accuracy: 0.9965\n",
            "Epoch 41/75\n",
            "36/36 [==============================] - 2s 62ms/step - loss: 0.0696 - accuracy: 0.9969\n",
            "Epoch 42/75\n",
            "36/36 [==============================] - 3s 83ms/step - loss: 0.0752 - accuracy: 0.9934\n",
            "Epoch 43/75\n",
            "36/36 [==============================] - 3s 74ms/step - loss: 0.0372 - accuracy: 0.9907\n",
            "Epoch 44/75\n",
            "36/36 [==============================] - 2s 62ms/step - loss: 0.1452 - accuracy: 0.9464\n",
            "Epoch 45/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.1369 - accuracy: 0.9460\n",
            "Epoch 46/75\n",
            "36/36 [==============================] - 2s 64ms/step - loss: 0.0411 - accuracy: 0.9876\n",
            "Epoch 47/75\n",
            "36/36 [==============================] - 2s 69ms/step - loss: 0.0246 - accuracy: 0.9925\n",
            "Epoch 48/75\n",
            "36/36 [==============================] - 3s 89ms/step - loss: 0.0199 - accuracy: 0.9929\n",
            "Epoch 49/75\n",
            "36/36 [==============================] - 2s 63ms/step - loss: 0.0156 - accuracy: 0.9965\n",
            "Epoch 50/75\n",
            "36/36 [==============================] - 2s 62ms/step - loss: 0.0132 - accuracy: 0.9969\n",
            "Epoch 51/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.0085 - accuracy: 0.9982\n",
            "Epoch 52/75\n",
            "36/36 [==============================] - 2s 62ms/step - loss: 0.0052 - accuracy: 0.9987\n",
            "Epoch 53/75\n",
            "36/36 [==============================] - 3s 94ms/step - loss: 0.0039 - accuracy: 0.9991\n",
            "Epoch 54/75\n",
            "36/36 [==============================] - 2s 65ms/step - loss: 0.0096 - accuracy: 0.9960\n",
            "Epoch 55/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.0058 - accuracy: 0.9978\n",
            "Epoch 56/75\n",
            "36/36 [==============================] - 2s 62ms/step - loss: 0.0099 - accuracy: 0.9973\n",
            "Epoch 57/75\n",
            "36/36 [==============================] - 2s 62ms/step - loss: 0.0090 - accuracy: 0.9969\n",
            "Epoch 58/75\n",
            "36/36 [==============================] - 3s 78ms/step - loss: 0.0043 - accuracy: 0.9991\n",
            "Epoch 59/75\n",
            "36/36 [==============================] - 3s 80ms/step - loss: 0.0064 - accuracy: 0.9978\n",
            "Epoch 60/75\n",
            "36/36 [==============================] - 2s 62ms/step - loss: 0.0054 - accuracy: 0.9987\n",
            "Epoch 61/75\n",
            "36/36 [==============================] - 2s 62ms/step - loss: 0.0043 - accuracy: 0.9982\n",
            "Epoch 62/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.0074 - accuracy: 0.9973\n",
            "Epoch 63/75\n",
            "36/36 [==============================] - 2s 64ms/step - loss: 0.0106 - accuracy: 0.9973\n",
            "Epoch 64/75\n",
            "36/36 [==============================] - 3s 93ms/step - loss: 0.0075 - accuracy: 0.9973\n",
            "Epoch 65/75\n",
            "36/36 [==============================] - 2s 62ms/step - loss: 0.0049 - accuracy: 0.9987\n",
            "Epoch 66/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.0130 - accuracy: 0.9960\n",
            "Epoch 67/75\n",
            "36/36 [==============================] - 2s 63ms/step - loss: 0.0086 - accuracy: 0.9969\n",
            "Epoch 68/75\n",
            "36/36 [==============================] - 2s 62ms/step - loss: 0.0112 - accuracy: 0.9960\n",
            "Epoch 69/75\n",
            "36/36 [==============================] - 3s 92ms/step - loss: 0.0222 - accuracy: 0.9929\n",
            "Epoch 70/75\n",
            "36/36 [==============================] - 2s 66ms/step - loss: 0.0767 - accuracy: 0.9761\n",
            "Epoch 71/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.0778 - accuracy: 0.9712\n",
            "Epoch 72/75\n",
            "36/36 [==============================] - 3s 72ms/step - loss: 0.0299 - accuracy: 0.9898\n",
            "Epoch 73/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.0127 - accuracy: 0.9951\n",
            "Epoch 74/75\n",
            "36/36 [==============================] - 3s 83ms/step - loss: 0.0054 - accuracy: 0.9991\n",
            "Epoch 75/75\n",
            "36/36 [==============================] - 3s 77ms/step - loss: 0.0036 - accuracy: 0.9991\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7e1f85f5cd00>"
            ]
          },
          "metadata": {},
          "execution_count": 92
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cnn.predict(xtest)\n",
        "pred = (pred > 0.5)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ONjW69jEC2lB",
        "outputId": "fc5e9826-cec4-4f62-c286-16956ec53b9f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "31/31 [==============================] - 0s 7ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(ytest, pred), precision_score(ytest, pred), recall_score(ytest, pred), f1_score(ytest, pred), cohen_kappa_score(ytest, pred), matthews_corrcoef(ytest, pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Xg8m8IIwC5tx",
        "outputId": "74c7d671-0e17-417e-8487-178ad6b875f9"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.932920536635707,\n",
              " 0.8178137651821862,\n",
              " 0.9099099099099099,\n",
              " 0.8614072494669509,\n",
              " 0.8173251777153249,\n",
              " 0.8193500009333097)"
            ]
          },
          "metadata": {},
          "execution_count": 94
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(ytest, pred)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ThUfVXSNC8P7",
        "outputId": "2989aaba-a3b0-4578-9f25-0674f431e2f2"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9397590361445783"
            ]
          },
          "metadata": {},
          "execution_count": 95
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**ADASYN**"
      ],
      "metadata": {
        "id": "lldVqUAAEAk_"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/NMB-TR.csv')"
      ],
      "metadata": {
        "id": "jSQIOiGhECcr"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "avBJPRbvEE7T"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.over_sampling import ADASYN\n",
        "ada = ADASYN()\n",
        "X, Y = ada.fit_resample(X, Y)"
      ],
      "metadata": {
        "id": "ePnYN6xZEHJr"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ],
      "metadata": {
        "id": "R9j2XwJZEJaj"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ],
      "metadata": {
        "id": "5RpzZK4WEMz7"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn = Sequential()"
      ],
      "metadata": {
        "id": "h7ZJr2xSEP9T"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))\n",
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu'))\n",
        "cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))"
      ],
      "metadata": {
        "id": "ynslnP-eESXL"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "bki_9Rr7EUL-"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Flatten())"
      ],
      "metadata": {
        "id": "bgVR-9q6EWUb"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "LAhfRjuBEYKq"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "dqk5exIvEaPe"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "63CAgG7DEcdL",
        "outputId": "7305d56c-481f-423b-9a49-524bf72cc9af"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "66/66 [==============================] - 6s 78ms/step - loss: 0.6650 - accuracy: 0.5900\n",
            "Epoch 2/75\n",
            "66/66 [==============================] - 8s 117ms/step - loss: 0.5848 - accuracy: 0.7099\n",
            "Epoch 3/75\n",
            "66/66 [==============================] - 6s 85ms/step - loss: 0.4908 - accuracy: 0.7740\n",
            "Epoch 4/75\n",
            "66/66 [==============================] - 6s 98ms/step - loss: 0.3743 - accuracy: 0.8436\n",
            "Epoch 5/75\n",
            "66/66 [==============================] - 5s 83ms/step - loss: 0.2757 - accuracy: 0.9015\n",
            "Epoch 6/75\n",
            "66/66 [==============================] - 5s 81ms/step - loss: 0.1987 - accuracy: 0.9390\n",
            "Epoch 7/75\n",
            "66/66 [==============================] - 6s 87ms/step - loss: 0.1455 - accuracy: 0.9573\n",
            "Epoch 8/75\n",
            "66/66 [==============================] - 8s 116ms/step - loss: 0.1207 - accuracy: 0.9604\n",
            "Epoch 9/75\n",
            "66/66 [==============================] - 6s 97ms/step - loss: 0.0882 - accuracy: 0.9709\n",
            "Epoch 10/75\n",
            "66/66 [==============================] - 7s 101ms/step - loss: 0.0578 - accuracy: 0.9838\n",
            "Epoch 11/75\n",
            "66/66 [==============================] - 6s 87ms/step - loss: 0.0640 - accuracy: 0.9793\n",
            "Epoch 12/75\n",
            "66/66 [==============================] - 7s 100ms/step - loss: 0.0528 - accuracy: 0.9840\n",
            "Epoch 13/75\n",
            "66/66 [==============================] - 7s 100ms/step - loss: 0.0229 - accuracy: 0.9943\n",
            "Epoch 14/75\n",
            "66/66 [==============================] - 7s 99ms/step - loss: 0.0599 - accuracy: 0.9824\n",
            "Epoch 15/75\n",
            "66/66 [==============================] - 6s 84ms/step - loss: 0.0451 - accuracy: 0.9859\n",
            "Epoch 16/75\n",
            "66/66 [==============================] - 6s 93ms/step - loss: 0.0149 - accuracy: 0.9969\n",
            "Epoch 17/75\n",
            "66/66 [==============================] - 6s 86ms/step - loss: 0.0088 - accuracy: 0.9988\n",
            "Epoch 18/75\n",
            "66/66 [==============================] - 6s 89ms/step - loss: 0.0169 - accuracy: 0.9969\n",
            "Epoch 19/75\n",
            "66/66 [==============================] - 7s 101ms/step - loss: 0.0107 - accuracy: 0.9979\n",
            "Epoch 20/75\n",
            "66/66 [==============================] - 9s 133ms/step - loss: 0.0222 - accuracy: 0.9928\n",
            "Epoch 21/75\n",
            "66/66 [==============================] - 6s 97ms/step - loss: 0.0458 - accuracy: 0.9874\n",
            "Epoch 22/75\n",
            "66/66 [==============================] - 7s 105ms/step - loss: 0.0454 - accuracy: 0.9847\n",
            "Epoch 23/75\n",
            "66/66 [==============================] - 7s 112ms/step - loss: 0.0131 - accuracy: 0.9974\n",
            "Epoch 24/75\n",
            "66/66 [==============================] - 7s 110ms/step - loss: 0.0060 - accuracy: 0.9988\n",
            "Epoch 25/75\n",
            "66/66 [==============================] - 6s 88ms/step - loss: 0.0056 - accuracy: 0.9990\n",
            "Epoch 26/75\n",
            "66/66 [==============================] - 7s 110ms/step - loss: 0.0049 - accuracy: 0.9993\n",
            "Epoch 27/75\n",
            "66/66 [==============================] - 6s 88ms/step - loss: 0.0171 - accuracy: 0.9969\n",
            "Epoch 28/75\n",
            "66/66 [==============================] - 6s 90ms/step - loss: 0.0249 - accuracy: 0.9924\n",
            "Epoch 29/75\n",
            "66/66 [==============================] - 5s 83ms/step - loss: 0.0173 - accuracy: 0.9957\n",
            "Epoch 30/75\n",
            "66/66 [==============================] - 6s 96ms/step - loss: 0.0069 - accuracy: 0.9983\n",
            "Epoch 31/75\n",
            "66/66 [==============================] - 5s 78ms/step - loss: 0.0066 - accuracy: 0.9990\n",
            "Epoch 32/75\n",
            "66/66 [==============================] - 4s 56ms/step - loss: 0.0050 - accuracy: 0.9990\n",
            "Epoch 33/75\n",
            "66/66 [==============================] - 5s 70ms/step - loss: 0.0090 - accuracy: 0.9981\n",
            "Epoch 34/75\n",
            "66/66 [==============================] - 4s 59ms/step - loss: 0.0057 - accuracy: 0.9990\n",
            "Epoch 35/75\n",
            "66/66 [==============================] - 4s 56ms/step - loss: 0.0076 - accuracy: 0.9981\n",
            "Epoch 36/75\n",
            "66/66 [==============================] - 4s 63ms/step - loss: 0.0084 - accuracy: 0.9981\n",
            "Epoch 37/75\n",
            "66/66 [==============================] - 4s 65ms/step - loss: 0.0366 - accuracy: 0.9912\n",
            "Epoch 38/75\n",
            "66/66 [==============================] - 4s 56ms/step - loss: 0.0735 - accuracy: 0.9754\n",
            "Epoch 39/75\n",
            "66/66 [==============================] - 4s 58ms/step - loss: 0.0456 - accuracy: 0.9843\n",
            "Epoch 40/75\n",
            "66/66 [==============================] - 5s 71ms/step - loss: 0.0159 - accuracy: 0.9967\n",
            "Epoch 41/75\n",
            "66/66 [==============================] - 4s 55ms/step - loss: 0.0061 - accuracy: 0.9990\n",
            "Epoch 42/75\n",
            "66/66 [==============================] - 4s 55ms/step - loss: 0.0114 - accuracy: 0.9979\n",
            "Epoch 43/75\n",
            "66/66 [==============================] - 5s 73ms/step - loss: 0.0168 - accuracy: 0.9952\n",
            "Epoch 44/75\n",
            "66/66 [==============================] - 4s 55ms/step - loss: 0.0121 - accuracy: 0.9962\n",
            "Epoch 45/75\n",
            "66/66 [==============================] - 4s 55ms/step - loss: 0.0072 - accuracy: 0.9990\n",
            "Epoch 46/75\n",
            "66/66 [==============================] - 6s 86ms/step - loss: 0.0120 - accuracy: 0.9967\n",
            "Epoch 47/75\n",
            "66/66 [==============================] - 7s 107ms/step - loss: 0.0073 - accuracy: 0.9981\n",
            "Epoch 48/75\n",
            "66/66 [==============================] - 8s 115ms/step - loss: 0.0053 - accuracy: 0.9988\n",
            "Epoch 49/75\n",
            "66/66 [==============================] - 6s 95ms/step - loss: 0.0055 - accuracy: 0.9988\n",
            "Epoch 50/75\n",
            "66/66 [==============================] - 8s 119ms/step - loss: 0.0034 - accuracy: 0.9988\n",
            "Epoch 51/75\n",
            "66/66 [==============================] - 6s 88ms/step - loss: 0.0205 - accuracy: 0.9940\n",
            "Epoch 52/75\n",
            "66/66 [==============================] - 7s 101ms/step - loss: 0.0479 - accuracy: 0.9838\n",
            "Epoch 53/75\n",
            "66/66 [==============================] - 6s 83ms/step - loss: 0.0269 - accuracy: 0.9909\n",
            "Epoch 54/75\n",
            "66/66 [==============================] - 7s 101ms/step - loss: 0.0097 - accuracy: 0.9981\n",
            "Epoch 55/75\n",
            "66/66 [==============================] - 6s 93ms/step - loss: 0.0084 - accuracy: 0.9974\n",
            "Epoch 56/75\n",
            "66/66 [==============================] - 8s 122ms/step - loss: 0.0055 - accuracy: 0.9981\n",
            "Epoch 57/75\n",
            "66/66 [==============================] - 5s 81ms/step - loss: 0.0059 - accuracy: 0.9988\n",
            "Epoch 58/75\n",
            "66/66 [==============================] - 7s 100ms/step - loss: 0.0043 - accuracy: 0.9993\n",
            "Epoch 59/75\n",
            "66/66 [==============================] - 5s 82ms/step - loss: 0.0045 - accuracy: 0.9993\n",
            "Epoch 60/75\n",
            "66/66 [==============================] - 7s 102ms/step - loss: 0.0040 - accuracy: 0.9990\n",
            "Epoch 61/75\n",
            "66/66 [==============================] - 6s 88ms/step - loss: 0.0027 - accuracy: 0.9993\n",
            "Epoch 62/75\n",
            "66/66 [==============================] - 6s 97ms/step - loss: 0.0033 - accuracy: 0.9993\n",
            "Epoch 63/75\n",
            "66/66 [==============================] - 6s 84ms/step - loss: 0.0040 - accuracy: 0.9993\n",
            "Epoch 64/75\n",
            "66/66 [==============================] - 6s 91ms/step - loss: 0.0037 - accuracy: 0.9993\n",
            "Epoch 65/75\n",
            "66/66 [==============================] - 6s 91ms/step - loss: 0.0052 - accuracy: 0.9988\n",
            "Epoch 66/75\n",
            "66/66 [==============================] - 7s 99ms/step - loss: 0.0036 - accuracy: 0.9993\n",
            "Epoch 67/75\n",
            "66/66 [==============================] - 9s 132ms/step - loss: 0.0038 - accuracy: 0.9986\n",
            "Epoch 68/75\n",
            "66/66 [==============================] - 9s 136ms/step - loss: 0.0033 - accuracy: 0.9993\n",
            "Epoch 69/75\n",
            "66/66 [==============================] - 8s 114ms/step - loss: 0.0056 - accuracy: 0.9983\n",
            "Epoch 70/75\n",
            "66/66 [==============================] - 7s 99ms/step - loss: 0.0037 - accuracy: 0.9995\n",
            "Epoch 71/75\n",
            "66/66 [==============================] - 5s 71ms/step - loss: 0.0043 - accuracy: 0.9993\n",
            "Epoch 72/75\n",
            "66/66 [==============================] - 5s 71ms/step - loss: 0.0036 - accuracy: 0.9993\n",
            "Epoch 73/75\n",
            "66/66 [==============================] - 4s 58ms/step - loss: 0.0051 - accuracy: 0.9986\n",
            "Epoch 74/75\n",
            "66/66 [==============================] - 4s 56ms/step - loss: 0.0059 - accuracy: 0.9986\n",
            "Epoch 75/75\n",
            "66/66 [==============================] - 4s 64ms/step - loss: 0.0056 - accuracy: 0.9988\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x794ccad121a0>"
            ]
          },
          "metadata": {},
          "execution_count": 52
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cnn.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "PDO1IxP1Eetz",
        "outputId": "279ffb1f-1550-4f59-d2e7-089185a3126e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "33/33 [==============================] - 0s 6ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "sq0CK2Q_EhBm",
        "outputId": "e65f5d54-4163-4dec-b364-4da427786ba5"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9751908396946565,\n",
              " 1.0,\n",
              " 0.9535714285714286,\n",
              " 0.9503165387869417,\n",
              " 0.9514916173794187)"
            ]
          },
          "metadata": {},
          "execution_count": 54
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MpfhAjIPEkOb",
        "outputId": "d697b856-d326-41cb-ef33-d5da872a7302"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9494163424124513"
            ]
          },
          "metadata": {},
          "execution_count": 55
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**SMOTETomek**"
      ],
      "metadata": {
        "id": "Ghcop5NkEnAD"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/NMB-TR.csv')"
      ],
      "metadata": {
        "id": "WmmAiLS9EqHb"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "8REGxo9XExon"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.combine import SMOTETomek\n",
        "smt = SMOTETomek()\n",
        "X, Y = smt.fit_resample(X, Y)"
      ],
      "metadata": {
        "id": "ojzHQyXXE047"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ],
      "metadata": {
        "id": "SetoTtvUE3by"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ],
      "metadata": {
        "id": "uIbPU62NE58k"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn = Sequential()"
      ],
      "metadata": {
        "id": "TL6WzKvOE8yx"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))\n",
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu'))\n",
        "cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))"
      ],
      "metadata": {
        "id": "1AOU_g8OE-66"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "4CgPlZjOFBqI"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Flatten())"
      ],
      "metadata": {
        "id": "shrLn_qqFD0r"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "3YwZvP-0FFtt"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "JrHzL0D_FHyi"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "VLO6sOiSFJxT",
        "outputId": "82c23f86-d313-4718-f970-f3ef499cbb33"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "65/65 [==============================] - 5s 56ms/step - loss: 0.6459 - accuracy: 0.6191\n",
            "Epoch 2/75\n",
            "65/65 [==============================] - 5s 73ms/step - loss: 0.5364 - accuracy: 0.7478\n",
            "Epoch 3/75\n",
            "65/65 [==============================] - 4s 56ms/step - loss: 0.4263 - accuracy: 0.8091\n",
            "Epoch 4/75\n",
            "65/65 [==============================] - 4s 55ms/step - loss: 0.3476 - accuracy: 0.8512\n",
            "Epoch 5/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.2663 - accuracy: 0.8916\n",
            "Epoch 6/75\n",
            "65/65 [==============================] - 4s 64ms/step - loss: 0.1897 - accuracy: 0.9276\n",
            "Epoch 7/75\n",
            "65/65 [==============================] - 4s 55ms/step - loss: 0.1466 - accuracy: 0.9497\n",
            "Epoch 8/75\n",
            "65/65 [==============================] - 4s 55ms/step - loss: 0.1215 - accuracy: 0.9572\n",
            "Epoch 9/75\n",
            "65/65 [==============================] - 5s 73ms/step - loss: 0.0710 - accuracy: 0.9789\n",
            "Epoch 10/75\n",
            "65/65 [==============================] - 4s 55ms/step - loss: 0.0595 - accuracy: 0.9818\n",
            "Epoch 11/75\n",
            "65/65 [==============================] - 4s 56ms/step - loss: 0.0579 - accuracy: 0.9835\n",
            "Epoch 12/75\n",
            "65/65 [==============================] - 5s 73ms/step - loss: 0.0475 - accuracy: 0.9831\n",
            "Epoch 13/75\n",
            "65/65 [==============================] - 4s 56ms/step - loss: 0.0455 - accuracy: 0.9852\n",
            "Epoch 14/75\n",
            "65/65 [==============================] - 4s 55ms/step - loss: 0.0226 - accuracy: 0.9947\n",
            "Epoch 15/75\n",
            "65/65 [==============================] - 5s 74ms/step - loss: 0.0152 - accuracy: 0.9964\n",
            "Epoch 16/75\n",
            "65/65 [==============================] - 4s 56ms/step - loss: 0.0176 - accuracy: 0.9949\n",
            "Epoch 17/75\n",
            "65/65 [==============================] - 4s 56ms/step - loss: 0.0474 - accuracy: 0.9862\n",
            "Epoch 18/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.0338 - accuracy: 0.9889\n",
            "Epoch 19/75\n",
            "65/65 [==============================] - 4s 64ms/step - loss: 0.0178 - accuracy: 0.9959\n",
            "Epoch 20/75\n",
            "65/65 [==============================] - 4s 56ms/step - loss: 0.0128 - accuracy: 0.9971\n",
            "Epoch 21/75\n",
            "65/65 [==============================] - 4s 58ms/step - loss: 0.0100 - accuracy: 0.9973\n",
            "Epoch 22/75\n",
            "65/65 [==============================] - 5s 70ms/step - loss: 0.0099 - accuracy: 0.9973\n",
            "Epoch 23/75\n",
            "65/65 [==============================] - 4s 55ms/step - loss: 0.0072 - accuracy: 0.9981\n",
            "Epoch 24/75\n",
            "65/65 [==============================] - 4s 56ms/step - loss: 0.0089 - accuracy: 0.9978\n",
            "Epoch 25/75\n",
            "65/65 [==============================] - 5s 73ms/step - loss: 0.0102 - accuracy: 0.9973\n",
            "Epoch 26/75\n",
            "65/65 [==============================] - 4s 56ms/step - loss: 0.0453 - accuracy: 0.9872\n",
            "Epoch 27/75\n",
            "65/65 [==============================] - 4s 55ms/step - loss: 0.0662 - accuracy: 0.9746\n",
            "Epoch 28/75\n",
            "65/65 [==============================] - 5s 73ms/step - loss: 0.0176 - accuracy: 0.9954\n",
            "Epoch 29/75\n",
            "65/65 [==============================] - 4s 55ms/step - loss: 0.0102 - accuracy: 0.9971\n",
            "Epoch 30/75\n",
            "65/65 [==============================] - 4s 56ms/step - loss: 0.0084 - accuracy: 0.9983\n",
            "Epoch 31/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.0067 - accuracy: 0.9978\n",
            "Epoch 32/75\n",
            "65/65 [==============================] - 4s 63ms/step - loss: 0.0071 - accuracy: 0.9981\n",
            "Epoch 33/75\n",
            "65/65 [==============================] - 4s 56ms/step - loss: 0.0060 - accuracy: 0.9983\n",
            "Epoch 34/75\n",
            "65/65 [==============================] - 4s 56ms/step - loss: 0.0046 - accuracy: 0.9985\n",
            "Epoch 35/75\n",
            "65/65 [==============================] - 5s 72ms/step - loss: 0.0085 - accuracy: 0.9981\n",
            "Epoch 36/75\n",
            "65/65 [==============================] - 4s 56ms/step - loss: 0.0150 - accuracy: 0.9964\n",
            "Epoch 37/75\n",
            "65/65 [==============================] - 4s 56ms/step - loss: 0.0061 - accuracy: 0.9993\n",
            "Epoch 38/75\n",
            "65/65 [==============================] - 5s 73ms/step - loss: 0.0047 - accuracy: 0.9988\n",
            "Epoch 39/75\n",
            "65/65 [==============================] - 4s 56ms/step - loss: 0.0204 - accuracy: 0.9939\n",
            "Epoch 40/75\n",
            "65/65 [==============================] - 4s 55ms/step - loss: 0.0180 - accuracy: 0.9942\n",
            "Epoch 41/75\n",
            "65/65 [==============================] - 5s 73ms/step - loss: 0.0149 - accuracy: 0.9954\n",
            "Epoch 42/75\n",
            "65/65 [==============================] - 4s 55ms/step - loss: 0.0083 - accuracy: 0.9971\n",
            "Epoch 43/75\n",
            "65/65 [==============================] - 4s 56ms/step - loss: 0.0163 - accuracy: 0.9949\n",
            "Epoch 44/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.0270 - accuracy: 0.9918\n",
            "Epoch 45/75\n",
            "65/65 [==============================] - 4s 63ms/step - loss: 0.0272 - accuracy: 0.9952\n",
            "Epoch 46/75\n",
            "65/65 [==============================] - 4s 56ms/step - loss: 0.0269 - accuracy: 0.9923\n",
            "Epoch 47/75\n",
            "65/65 [==============================] - 4s 57ms/step - loss: 0.0076 - accuracy: 0.9976\n",
            "Epoch 48/75\n",
            "65/65 [==============================] - 5s 72ms/step - loss: 0.0062 - accuracy: 0.9988\n",
            "Epoch 49/75\n",
            "65/65 [==============================] - 4s 56ms/step - loss: 0.0053 - accuracy: 0.9985\n",
            "Epoch 50/75\n",
            "65/65 [==============================] - 4s 56ms/step - loss: 0.0037 - accuracy: 0.9988\n",
            "Epoch 51/75\n",
            "65/65 [==============================] - 5s 73ms/step - loss: 0.0046 - accuracy: 0.9990\n",
            "Epoch 52/75\n",
            "65/65 [==============================] - 4s 55ms/step - loss: 0.0074 - accuracy: 0.9985\n",
            "Epoch 53/75\n",
            "65/65 [==============================] - 4s 55ms/step - loss: 0.0041 - accuracy: 0.9988\n",
            "Epoch 54/75\n",
            "65/65 [==============================] - 5s 73ms/step - loss: 0.0034 - accuracy: 0.9995\n",
            "Epoch 55/75\n",
            "65/65 [==============================] - 4s 55ms/step - loss: 0.0061 - accuracy: 0.9983\n",
            "Epoch 56/75\n",
            "65/65 [==============================] - 4s 55ms/step - loss: 0.0059 - accuracy: 0.9981\n",
            "Epoch 57/75\n",
            "65/65 [==============================] - 4s 66ms/step - loss: 0.0038 - accuracy: 0.9983\n",
            "Epoch 58/75\n",
            "65/65 [==============================] - 4s 62ms/step - loss: 0.0078 - accuracy: 0.9983\n",
            "Epoch 59/75\n",
            "65/65 [==============================] - 4s 56ms/step - loss: 0.0045 - accuracy: 0.9985\n",
            "Epoch 60/75\n",
            "65/65 [==============================] - 4s 58ms/step - loss: 0.0027 - accuracy: 0.9993\n",
            "Epoch 61/75\n",
            "65/65 [==============================] - 5s 71ms/step - loss: 0.0056 - accuracy: 0.9985\n",
            "Epoch 62/75\n",
            "65/65 [==============================] - 4s 55ms/step - loss: 0.0053 - accuracy: 0.9993\n",
            "Epoch 63/75\n",
            "65/65 [==============================] - 4s 56ms/step - loss: 0.0034 - accuracy: 0.9990\n",
            "Epoch 64/75\n",
            "65/65 [==============================] - 5s 73ms/step - loss: 0.0035 - accuracy: 0.9993\n",
            "Epoch 65/75\n",
            "65/65 [==============================] - 4s 55ms/step - loss: 0.0032 - accuracy: 0.9988\n",
            "Epoch 66/75\n",
            "65/65 [==============================] - 4s 56ms/step - loss: 0.0024 - accuracy: 0.9995\n",
            "Epoch 67/75\n",
            "65/65 [==============================] - 5s 74ms/step - loss: 0.0144 - accuracy: 0.9964\n",
            "Epoch 68/75\n",
            "65/65 [==============================] - 4s 57ms/step - loss: 0.0309 - accuracy: 0.9906\n",
            "Epoch 69/75\n",
            "65/65 [==============================] - 4s 56ms/step - loss: 0.1163 - accuracy: 0.9618\n",
            "Epoch 70/75\n",
            "65/65 [==============================] - 4s 69ms/step - loss: 0.0241 - accuracy: 0.9927\n",
            "Epoch 71/75\n",
            "65/65 [==============================] - 4s 60ms/step - loss: 0.0078 - accuracy: 0.9983\n",
            "Epoch 72/75\n",
            "65/65 [==============================] - 4s 56ms/step - loss: 0.0082 - accuracy: 0.9978\n",
            "Epoch 73/75\n",
            "65/65 [==============================] - 4s 61ms/step - loss: 0.0055 - accuracy: 0.9988\n",
            "Epoch 74/75\n",
            "65/65 [==============================] - 4s 69ms/step - loss: 0.0041 - accuracy: 0.9990\n",
            "Epoch 75/75\n",
            "65/65 [==============================] - 4s 57ms/step - loss: 0.0034 - accuracy: 0.9993\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x794ccab9d060>"
            ]
          },
          "metadata": {},
          "execution_count": 67
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cnn.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_QTq-ZB1FMQS",
        "outputId": "e36b5931-1b73-422f-91b7-66b9b755f07c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "33/33 [==============================] - 0s 7ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-RaI9o8eFPFS",
        "outputId": "76efc2a5-6b4b-4320-b0de-3113a6db9cd4"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9777131782945736,\n",
              " 1.0,\n",
              " 0.9587073608617595,\n",
              " 0.9553024653695794,\n",
              " 0.9562581825351)"
            ]
          },
          "metadata": {},
          "execution_count": 69
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "LQqHbNyDFSva",
        "outputId": "68b835fe-2f60-4dd2-a350-2b67b4027a98"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9538152610441767"
            ]
          },
          "metadata": {},
          "execution_count": 70
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**NearMiss**"
      ],
      "metadata": {
        "id": "yvnniqxuFU8C"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/NMB-TR.csv')"
      ],
      "metadata": {
        "id": "3cumaOkmFZA6"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "U7mQ9a0PFtma"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.under_sampling import NearMiss\n",
        "nm = NearMiss()\n",
        "X, Y = nm.fit_resample(X, Y)"
      ],
      "metadata": {
        "id": "XA93132QFv4v"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ],
      "metadata": {
        "id": "La7HF0JCFyCy"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ],
      "metadata": {
        "id": "wscYKc2bF0I6"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn = Sequential()"
      ],
      "metadata": {
        "id": "pqhB5SnuF17s"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))\n",
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu'))\n",
        "cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))"
      ],
      "metadata": {
        "id": "-6JbK5kPF32K"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "Ch9Yowg8F6Cy"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Flatten())"
      ],
      "metadata": {
        "id": "0WOsQDR7F8F_"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "X0Fgo7YyF-Dy"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "JkpWDzs-GAGO"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "KJ5I527WGC_k",
        "outputId": "7f0476ba-387e-4fb2-cd39-06b2090789cd"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "17/17 [==============================] - 2s 54ms/step - loss: 0.6428 - accuracy: 0.5242\n",
            "Epoch 2/75\n",
            "17/17 [==============================] - 1s 56ms/step - loss: 0.5427 - accuracy: 0.7248\n",
            "Epoch 3/75\n",
            "17/17 [==============================] - 1s 86ms/step - loss: 0.4725 - accuracy: 0.7578\n",
            "Epoch 4/75\n",
            "17/17 [==============================] - 1s 85ms/step - loss: 0.5177 - accuracy: 0.7374\n",
            "Epoch 5/75\n",
            "17/17 [==============================] - 1s 54ms/step - loss: 0.4571 - accuracy: 0.7752\n",
            "Epoch 6/75\n",
            "17/17 [==============================] - 1s 53ms/step - loss: 0.4252 - accuracy: 0.7694\n",
            "Epoch 7/75\n",
            "17/17 [==============================] - 1s 54ms/step - loss: 0.4241 - accuracy: 0.7955\n",
            "Epoch 8/75\n",
            "17/17 [==============================] - 1s 54ms/step - loss: 0.3985 - accuracy: 0.8004\n",
            "Epoch 9/75\n",
            "17/17 [==============================] - 1s 53ms/step - loss: 0.3844 - accuracy: 0.8140\n",
            "Epoch 10/75\n",
            "17/17 [==============================] - 1s 54ms/step - loss: 0.3939 - accuracy: 0.8110\n",
            "Epoch 11/75\n",
            "17/17 [==============================] - 1s 54ms/step - loss: 0.3735 - accuracy: 0.8256\n",
            "Epoch 12/75\n",
            "17/17 [==============================] - 1s 55ms/step - loss: 0.3354 - accuracy: 0.8566\n",
            "Epoch 13/75\n",
            "17/17 [==============================] - 1s 53ms/step - loss: 0.3199 - accuracy: 0.8643\n",
            "Epoch 14/75\n",
            "17/17 [==============================] - 1s 54ms/step - loss: 0.3004 - accuracy: 0.8808\n",
            "Epoch 15/75\n",
            "17/17 [==============================] - 1s 61ms/step - loss: 0.2913 - accuracy: 0.8915\n",
            "Epoch 16/75\n",
            "17/17 [==============================] - 2s 89ms/step - loss: 0.2581 - accuracy: 0.9031\n",
            "Epoch 17/75\n",
            "17/17 [==============================] - 1s 77ms/step - loss: 0.2411 - accuracy: 0.9099\n",
            "Epoch 18/75\n",
            "17/17 [==============================] - 1s 53ms/step - loss: 0.2436 - accuracy: 0.9079\n",
            "Epoch 19/75\n",
            "17/17 [==============================] - 1s 53ms/step - loss: 0.3137 - accuracy: 0.8731\n",
            "Epoch 20/75\n",
            "17/17 [==============================] - 1s 54ms/step - loss: 0.2540 - accuracy: 0.9021\n",
            "Epoch 21/75\n",
            "17/17 [==============================] - 1s 54ms/step - loss: 0.2097 - accuracy: 0.9215\n",
            "Epoch 22/75\n",
            "17/17 [==============================] - 1s 53ms/step - loss: 0.1820 - accuracy: 0.9409\n",
            "Epoch 23/75\n",
            "17/17 [==============================] - 1s 53ms/step - loss: 0.2137 - accuracy: 0.9234\n",
            "Epoch 24/75\n",
            "17/17 [==============================] - 1s 54ms/step - loss: 0.1736 - accuracy: 0.9428\n",
            "Epoch 25/75\n",
            "17/17 [==============================] - 1s 53ms/step - loss: 0.1501 - accuracy: 0.9545\n",
            "Epoch 26/75\n",
            "17/17 [==============================] - 1s 53ms/step - loss: 0.1763 - accuracy: 0.9448\n",
            "Epoch 27/75\n",
            "17/17 [==============================] - 1s 54ms/step - loss: 0.1352 - accuracy: 0.9622\n",
            "Epoch 28/75\n",
            "17/17 [==============================] - 1s 62ms/step - loss: 0.1062 - accuracy: 0.9680\n",
            "Epoch 29/75\n",
            "17/17 [==============================] - 2s 88ms/step - loss: 0.0919 - accuracy: 0.9748\n",
            "Epoch 30/75\n",
            "17/17 [==============================] - 1s 75ms/step - loss: 0.0824 - accuracy: 0.9806\n",
            "Epoch 31/75\n",
            "17/17 [==============================] - 1s 53ms/step - loss: 0.0796 - accuracy: 0.9787\n",
            "Epoch 32/75\n",
            "17/17 [==============================] - 1s 52ms/step - loss: 0.0871 - accuracy: 0.9738\n",
            "Epoch 33/75\n",
            "17/17 [==============================] - 1s 54ms/step - loss: 0.0708 - accuracy: 0.9816\n",
            "Epoch 34/75\n",
            "17/17 [==============================] - 1s 54ms/step - loss: 0.2284 - accuracy: 0.9128\n",
            "Epoch 35/75\n",
            "17/17 [==============================] - 1s 54ms/step - loss: 0.1346 - accuracy: 0.9583\n",
            "Epoch 36/75\n",
            "17/17 [==============================] - 1s 54ms/step - loss: 0.0979 - accuracy: 0.9700\n",
            "Epoch 37/75\n",
            "17/17 [==============================] - 1s 54ms/step - loss: 0.0715 - accuracy: 0.9806\n",
            "Epoch 38/75\n",
            "17/17 [==============================] - 1s 54ms/step - loss: 0.0765 - accuracy: 0.9709\n",
            "Epoch 39/75\n",
            "17/17 [==============================] - 1s 53ms/step - loss: 0.0725 - accuracy: 0.9767\n",
            "Epoch 40/75\n",
            "17/17 [==============================] - 1s 53ms/step - loss: 0.0767 - accuracy: 0.9758\n",
            "Epoch 41/75\n",
            "17/17 [==============================] - 1s 71ms/step - loss: 0.0390 - accuracy: 0.9932\n",
            "Epoch 42/75\n",
            "17/17 [==============================] - 1s 88ms/step - loss: 0.0303 - accuracy: 0.9903\n",
            "Epoch 43/75\n",
            "17/17 [==============================] - 1s 68ms/step - loss: 0.0323 - accuracy: 0.9884\n",
            "Epoch 44/75\n",
            "17/17 [==============================] - 1s 55ms/step - loss: 0.0289 - accuracy: 0.9903\n",
            "Epoch 45/75\n",
            "17/17 [==============================] - 1s 53ms/step - loss: 0.0273 - accuracy: 0.9913\n",
            "Epoch 46/75\n",
            "17/17 [==============================] - 1s 55ms/step - loss: 0.0226 - accuracy: 0.9952\n",
            "Epoch 47/75\n",
            "17/17 [==============================] - 1s 54ms/step - loss: 0.0201 - accuracy: 0.9952\n",
            "Epoch 48/75\n",
            "17/17 [==============================] - 1s 54ms/step - loss: 0.0212 - accuracy: 0.9952\n",
            "Epoch 49/75\n",
            "17/17 [==============================] - 1s 54ms/step - loss: 0.0246 - accuracy: 0.9932\n",
            "Epoch 50/75\n",
            "17/17 [==============================] - 1s 55ms/step - loss: 0.0222 - accuracy: 0.9942\n",
            "Epoch 51/75\n",
            "17/17 [==============================] - 1s 53ms/step - loss: 0.0257 - accuracy: 0.9942\n",
            "Epoch 52/75\n",
            "17/17 [==============================] - 1s 54ms/step - loss: 0.0270 - accuracy: 0.9913\n",
            "Epoch 53/75\n",
            "17/17 [==============================] - 1s 53ms/step - loss: 0.0226 - accuracy: 0.9913\n",
            "Epoch 54/75\n",
            "17/17 [==============================] - 1s 79ms/step - loss: 0.0292 - accuracy: 0.9903\n",
            "Epoch 55/75\n",
            "17/17 [==============================] - 1s 87ms/step - loss: 0.0292 - accuracy: 0.9922\n",
            "Epoch 56/75\n",
            "17/17 [==============================] - 1s 60ms/step - loss: 0.0129 - accuracy: 0.9961\n",
            "Epoch 57/75\n",
            "17/17 [==============================] - 1s 54ms/step - loss: 0.0170 - accuracy: 0.9942\n",
            "Epoch 58/75\n",
            "17/17 [==============================] - 1s 54ms/step - loss: 0.0253 - accuracy: 0.9932\n",
            "Epoch 59/75\n",
            "17/17 [==============================] - 1s 54ms/step - loss: 0.0232 - accuracy: 0.9913\n",
            "Epoch 60/75\n",
            "17/17 [==============================] - 1s 54ms/step - loss: 0.0305 - accuracy: 0.9922\n",
            "Epoch 61/75\n",
            "17/17 [==============================] - 1s 54ms/step - loss: 0.0275 - accuracy: 0.9913\n",
            "Epoch 62/75\n",
            "17/17 [==============================] - 1s 54ms/step - loss: 0.0237 - accuracy: 0.9961\n",
            "Epoch 63/75\n",
            "17/17 [==============================] - 1s 54ms/step - loss: 0.0400 - accuracy: 0.9903\n",
            "Epoch 64/75\n",
            "17/17 [==============================] - 1s 54ms/step - loss: 0.0182 - accuracy: 0.9961\n",
            "Epoch 65/75\n",
            "17/17 [==============================] - 1s 54ms/step - loss: 0.0129 - accuracy: 0.9952\n",
            "Epoch 66/75\n",
            "17/17 [==============================] - 1s 55ms/step - loss: 0.0134 - accuracy: 0.9961\n",
            "Epoch 67/75\n",
            "17/17 [==============================] - 1s 88ms/step - loss: 0.0149 - accuracy: 0.9932\n",
            "Epoch 68/75\n",
            "17/17 [==============================] - 1s 84ms/step - loss: 0.0167 - accuracy: 0.9952\n",
            "Epoch 69/75\n",
            "17/17 [==============================] - 1s 53ms/step - loss: 0.0156 - accuracy: 0.9961\n",
            "Epoch 70/75\n",
            "17/17 [==============================] - 1s 54ms/step - loss: 0.0150 - accuracy: 0.9942\n",
            "Epoch 71/75\n",
            "17/17 [==============================] - 1s 54ms/step - loss: 0.0135 - accuracy: 0.9952\n",
            "Epoch 72/75\n",
            "17/17 [==============================] - 1s 53ms/step - loss: 0.0099 - accuracy: 0.9961\n",
            "Epoch 73/75\n",
            "17/17 [==============================] - 1s 53ms/step - loss: 0.0129 - accuracy: 0.9961\n",
            "Epoch 74/75\n",
            "17/17 [==============================] - 1s 54ms/step - loss: 0.0106 - accuracy: 0.9961\n",
            "Epoch 75/75\n",
            "17/17 [==============================] - 1s 53ms/step - loss: 0.0132 - accuracy: 0.9952\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x794ccad133d0>"
            ]
          },
          "metadata": {},
          "execution_count": 82
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cnn.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "1vIzBEZiGFg6",
        "outputId": "edb68788-c12b-41ae-e989-7a15d876412d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "9/9 [==============================] - 0s 6ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "D-EKiFtfGIg7",
        "outputId": "680ad515-8075-414d-819d-b0a1dac2d0ae"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9457364341085271,\n",
              " 1.0,\n",
              " 0.9020979020979021,\n",
              " 0.8914728682170543,\n",
              " 0.8967696494617802)"
            ]
          },
          "metadata": {},
          "execution_count": 84
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3Gjb3KIrGKqq",
        "outputId": "abfd9990-3c7b-4977-9793-bc3b2328ccfd"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.8914728682170543"
            ]
          },
          "metadata": {},
          "execution_count": 85
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **ANN(LSA)**"
      ],
      "metadata": {
        "id": "bZQ5oWq7JLDW"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Imbalanced**"
      ],
      "metadata": {
        "id": "pyk_GBLnKlg1"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/LSA_TR.csv')"
      ],
      "metadata": {
        "id": "X42j5-LDJPJY"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "u_x2hKFjJZHh"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ],
      "metadata": {
        "id": "iYnYnuL0Jbjw"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ],
      "metadata": {
        "id": "eq9z4gnQJdlL"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann = Sequential()"
      ],
      "metadata": {
        "id": "cYJ9YEdAJfsI"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Dense(256, activation = 'relu', input_shape=(X_train.shape[1], 1)))\n",
        "ann.add(Dense(256, activation = 'relu'))\n",
        "ann.add(Dense(128, activation = 'relu'))"
      ],
      "metadata": {
        "id": "GCDtV1F7Jkhx"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "Sv9-VRcjKHw_"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Flatten())"
      ],
      "metadata": {
        "id": "G5pjd6QTKNMh"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Dense(64, activation='relu'))\n",
        "ann.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "CinQG5xDJ_eA"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "v-QJUBa8KQXZ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "k9drLuMLKTgz",
        "outputId": "630deae2-377c-4688-afb9-fee722bf3d6c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "41/41 [==============================] - 4s 52ms/step - loss: 0.5322 - accuracy: 0.7909\n",
            "Epoch 2/75\n",
            "41/41 [==============================] - 3s 83ms/step - loss: 0.4780 - accuracy: 0.8005\n",
            "Epoch 3/75\n",
            "41/41 [==============================] - 5s 112ms/step - loss: 0.3986 - accuracy: 0.8338\n",
            "Epoch 4/75\n",
            "41/41 [==============================] - 4s 95ms/step - loss: 0.3577 - accuracy: 0.8563\n",
            "Epoch 5/75\n",
            "41/41 [==============================] - 4s 92ms/step - loss: 0.3397 - accuracy: 0.8633\n",
            "Epoch 6/75\n",
            "41/41 [==============================] - 5s 113ms/step - loss: 0.3027 - accuracy: 0.8768\n",
            "Epoch 7/75\n",
            "41/41 [==============================] - 4s 89ms/step - loss: 0.2748 - accuracy: 0.8811\n",
            "Epoch 8/75\n",
            "41/41 [==============================] - 4s 93ms/step - loss: 0.2667 - accuracy: 0.8850\n",
            "Epoch 9/75\n",
            "41/41 [==============================] - 4s 85ms/step - loss: 0.2287 - accuracy: 0.9040\n",
            "Epoch 10/75\n",
            "41/41 [==============================] - 3s 66ms/step - loss: 0.2170 - accuracy: 0.9067\n",
            "Epoch 11/75\n",
            "41/41 [==============================] - 2s 53ms/step - loss: 0.2104 - accuracy: 0.9090\n",
            "Epoch 12/75\n",
            "41/41 [==============================] - 2s 53ms/step - loss: 0.1906 - accuracy: 0.9163\n",
            "Epoch 13/75\n",
            "41/41 [==============================] - 2s 53ms/step - loss: 0.1826 - accuracy: 0.9218\n",
            "Epoch 14/75\n",
            "41/41 [==============================] - 4s 89ms/step - loss: 0.1705 - accuracy: 0.9322\n",
            "Epoch 15/75\n",
            "41/41 [==============================] - 4s 101ms/step - loss: 0.1528 - accuracy: 0.9404\n",
            "Epoch 16/75\n",
            "41/41 [==============================] - 4s 89ms/step - loss: 0.1595 - accuracy: 0.9353\n",
            "Epoch 17/75\n",
            "41/41 [==============================] - 4s 94ms/step - loss: 0.1479 - accuracy: 0.9435\n",
            "Epoch 18/75\n",
            "41/41 [==============================] - 5s 115ms/step - loss: 0.1303 - accuracy: 0.9481\n",
            "Epoch 19/75\n",
            "41/41 [==============================] - 3s 68ms/step - loss: 0.1146 - accuracy: 0.9570\n",
            "Epoch 20/75\n",
            "41/41 [==============================] - 3s 84ms/step - loss: 0.0976 - accuracy: 0.9671\n",
            "Epoch 21/75\n",
            "41/41 [==============================] - 4s 108ms/step - loss: 0.0934 - accuracy: 0.9648\n",
            "Epoch 22/75\n",
            "41/41 [==============================] - 3s 80ms/step - loss: 0.0745 - accuracy: 0.9744\n",
            "Epoch 23/75\n",
            "41/41 [==============================] - 4s 95ms/step - loss: 0.0708 - accuracy: 0.9795\n",
            "Epoch 24/75\n",
            "41/41 [==============================] - 4s 91ms/step - loss: 0.0647 - accuracy: 0.9771\n",
            "Epoch 25/75\n",
            "41/41 [==============================] - 4s 106ms/step - loss: 0.0391 - accuracy: 0.9915\n",
            "Epoch 26/75\n",
            "41/41 [==============================] - 5s 128ms/step - loss: 0.0306 - accuracy: 0.9923\n",
            "Epoch 27/75\n",
            "41/41 [==============================] - 6s 153ms/step - loss: 0.0242 - accuracy: 0.9954\n",
            "Epoch 28/75\n",
            "41/41 [==============================] - 4s 109ms/step - loss: 0.0172 - accuracy: 0.9977\n",
            "Epoch 29/75\n",
            "41/41 [==============================] - 5s 128ms/step - loss: 0.0162 - accuracy: 0.9961\n",
            "Epoch 30/75\n",
            "41/41 [==============================] - 6s 144ms/step - loss: 0.0099 - accuracy: 0.9988\n",
            "Epoch 31/75\n",
            "41/41 [==============================] - 4s 106ms/step - loss: 0.0068 - accuracy: 1.0000\n",
            "Epoch 32/75\n",
            "41/41 [==============================] - 5s 130ms/step - loss: 0.0050 - accuracy: 1.0000\n",
            "Epoch 33/75\n",
            "41/41 [==============================] - 4s 106ms/step - loss: 0.0040 - accuracy: 1.0000\n",
            "Epoch 34/75\n",
            "41/41 [==============================] - 4s 93ms/step - loss: 0.0031 - accuracy: 1.0000\n",
            "Epoch 35/75\n",
            "41/41 [==============================] - 5s 111ms/step - loss: 0.0023 - accuracy: 1.0000\n",
            "Epoch 36/75\n",
            "41/41 [==============================] - 4s 104ms/step - loss: 0.0020 - accuracy: 1.0000\n",
            "Epoch 37/75\n",
            "41/41 [==============================] - 4s 97ms/step - loss: 0.0017 - accuracy: 1.0000\n",
            "Epoch 38/75\n",
            "41/41 [==============================] - 5s 127ms/step - loss: 0.0014 - accuracy: 1.0000\n",
            "Epoch 39/75\n",
            "41/41 [==============================] - 5s 112ms/step - loss: 0.0013 - accuracy: 1.0000\n",
            "Epoch 40/75\n",
            "41/41 [==============================] - 5s 130ms/step - loss: 0.0012 - accuracy: 1.0000\n",
            "Epoch 41/75\n",
            "41/41 [==============================] - 5s 116ms/step - loss: 9.8290e-04 - accuracy: 1.0000\n",
            "Epoch 42/75\n",
            "41/41 [==============================] - 5s 114ms/step - loss: 8.8784e-04 - accuracy: 1.0000\n",
            "Epoch 43/75\n",
            "41/41 [==============================] - 4s 105ms/step - loss: 8.1040e-04 - accuracy: 1.0000\n",
            "Epoch 44/75\n",
            "41/41 [==============================] - 6s 135ms/step - loss: 7.1891e-04 - accuracy: 1.0000\n",
            "Epoch 45/75\n",
            "41/41 [==============================] - 4s 89ms/step - loss: 6.9314e-04 - accuracy: 1.0000\n",
            "Epoch 46/75\n",
            "41/41 [==============================] - 4s 91ms/step - loss: 7.3484e-04 - accuracy: 1.0000\n",
            "Epoch 47/75\n",
            "41/41 [==============================] - 5s 115ms/step - loss: 5.7160e-04 - accuracy: 1.0000\n",
            "Epoch 48/75\n",
            "41/41 [==============================] - 2s 54ms/step - loss: 5.1494e-04 - accuracy: 1.0000\n",
            "Epoch 49/75\n",
            "41/41 [==============================] - 2s 53ms/step - loss: 4.6582e-04 - accuracy: 1.0000\n",
            "Epoch 50/75\n",
            "41/41 [==============================] - 2s 53ms/step - loss: 4.3830e-04 - accuracy: 1.0000\n",
            "Epoch 51/75\n",
            "41/41 [==============================] - 4s 91ms/step - loss: 4.1038e-04 - accuracy: 1.0000\n",
            "Epoch 52/75\n",
            "41/41 [==============================] - 5s 108ms/step - loss: 3.7814e-04 - accuracy: 1.0000\n",
            "Epoch 53/75\n",
            "41/41 [==============================] - 4s 97ms/step - loss: 3.5178e-04 - accuracy: 1.0000\n",
            "Epoch 54/75\n",
            "41/41 [==============================] - 3s 82ms/step - loss: 3.3584e-04 - accuracy: 1.0000\n",
            "Epoch 55/75\n",
            "41/41 [==============================] - 3s 80ms/step - loss: 3.1742e-04 - accuracy: 1.0000\n",
            "Epoch 56/75\n",
            "41/41 [==============================] - 3s 79ms/step - loss: 2.9552e-04 - accuracy: 1.0000\n",
            "Epoch 57/75\n",
            "41/41 [==============================] - 4s 97ms/step - loss: 2.7904e-04 - accuracy: 1.0000\n",
            "Epoch 58/75\n",
            "41/41 [==============================] - 6s 143ms/step - loss: 2.5681e-04 - accuracy: 1.0000\n",
            "Epoch 59/75\n",
            "41/41 [==============================] - 4s 102ms/step - loss: 2.4265e-04 - accuracy: 1.0000\n",
            "Epoch 60/75\n",
            "41/41 [==============================] - 4s 97ms/step - loss: 2.2901e-04 - accuracy: 1.0000\n",
            "Epoch 61/75\n",
            "41/41 [==============================] - 6s 143ms/step - loss: 2.1553e-04 - accuracy: 1.0000\n",
            "Epoch 62/75\n",
            "41/41 [==============================] - 4s 105ms/step - loss: 2.0564e-04 - accuracy: 1.0000\n",
            "Epoch 63/75\n",
            "41/41 [==============================] - 4s 101ms/step - loss: 1.9315e-04 - accuracy: 1.0000\n",
            "Epoch 64/75\n",
            "41/41 [==============================] - 4s 108ms/step - loss: 1.8479e-04 - accuracy: 1.0000\n",
            "Epoch 65/75\n",
            "41/41 [==============================] - 4s 94ms/step - loss: 1.7655e-04 - accuracy: 1.0000\n",
            "Epoch 66/75\n",
            "41/41 [==============================] - 4s 97ms/step - loss: 1.6712e-04 - accuracy: 1.0000\n",
            "Epoch 67/75\n",
            "41/41 [==============================] - 5s 124ms/step - loss: 1.5912e-04 - accuracy: 1.0000\n",
            "Epoch 68/75\n",
            "41/41 [==============================] - 5s 110ms/step - loss: 1.5234e-04 - accuracy: 1.0000\n",
            "Epoch 69/75\n",
            "41/41 [==============================] - 6s 152ms/step - loss: 1.4519e-04 - accuracy: 1.0000\n",
            "Epoch 70/75\n",
            "41/41 [==============================] - 5s 112ms/step - loss: 1.3700e-04 - accuracy: 1.0000\n",
            "Epoch 71/75\n",
            "41/41 [==============================] - 4s 96ms/step - loss: 1.3471e-04 - accuracy: 1.0000\n",
            "Epoch 72/75\n",
            "41/41 [==============================] - 4s 110ms/step - loss: 1.2885e-04 - accuracy: 1.0000\n",
            "Epoch 73/75\n",
            "41/41 [==============================] - 3s 82ms/step - loss: 1.2139e-04 - accuracy: 1.0000\n",
            "Epoch 74/75\n",
            "41/41 [==============================] - 4s 89ms/step - loss: 1.2240e-04 - accuracy: 1.0000\n",
            "Epoch 75/75\n",
            "41/41 [==============================] - 4s 91ms/step - loss: 1.1083e-04 - accuracy: 1.0000\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x794cc2b45300>"
            ]
          },
          "metadata": {},
          "execution_count": 98
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = ann.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "f6MOtwByKXl4",
        "outputId": "8570fc67-a943-44b8-fa07-86ce778e5ccc"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "21/21 [==============================] - 1s 16ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "9oFGnmPzKaZ_",
        "outputId": "5e39d41f-3460-430d-8e01-680dd9a67c6c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9627906976744186,\n",
              " 0.9694656488549618,\n",
              " 0.8639455782312925,\n",
              " 0.890053694707236,\n",
              " 0.8924542916099305)"
            ]
          },
          "metadata": {},
          "execution_count": 101
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_r4r9bL2KiLS",
        "outputId": "7b4c138a-cfc7-4c71-fd76-35a5294f3b0c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9610894941634242"
            ]
          },
          "metadata": {},
          "execution_count": 102
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Test**"
      ],
      "metadata": {
        "id": "sochXjrS5Bxk"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/LSA_TR.csv')\n",
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "kPKayP695KFy"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.3, random_state = 1)"
      ],
      "metadata": {
        "id": "-QuLW4JinoiD"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "sample_size = xtrain.shape[0] # number of samples in train set\n",
        "time_steps  = xtrain.shape[1] # number of features in train set\n",
        "input_dimension = 1               # each feature is represented by 1 number"
      ],
      "metadata": {
        "id": "0Dta01uy5g_J"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_data_reshaped = xtrain.values.reshape(sample_size,time_steps,input_dimension)\n",
        "n_timesteps = train_data_reshaped.shape[1]\n",
        "n_features  = train_data_reshaped.shape[2]"
      ],
      "metadata": {
        "id": "dc34afh45kts"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann = Sequential()"
      ],
      "metadata": {
        "id": "NekBlWo35m9y"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Dense(256, activation = 'relu', input_shape=(n_timesteps,n_features)))\n",
        "ann.add(Dense(256, activation = 'relu'))\n",
        "ann.add(Dense(128, activation = 'relu'))"
      ],
      "metadata": {
        "id": "006hTtS_5o1i"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "7Igx75TY5rb6"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Flatten())"
      ],
      "metadata": {
        "id": "Akd-Pan55tXq"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Dense(64, activation='relu'))\n",
        "ann.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "9ybPETWV5vPE"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "KPk_sgsU5xqq"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.fit(xtrain, ytrain, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "2L6H3sxW5z_L",
        "outputId": "eb2698e7-ba4e-4a5e-ad8f-e09bad55d744"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "36/36 [==============================] - 11s 120ms/step - loss: 0.5315 - accuracy: 0.7958\n",
            "Epoch 2/75\n",
            "36/36 [==============================] - 4s 116ms/step - loss: 0.4819 - accuracy: 0.8127\n",
            "Epoch 3/75\n",
            "36/36 [==============================] - 6s 154ms/step - loss: 0.4716 - accuracy: 0.8127\n",
            "Epoch 4/75\n",
            "36/36 [==============================] - 4s 123ms/step - loss: 0.4005 - accuracy: 0.8224\n",
            "Epoch 5/75\n",
            "36/36 [==============================] - 4s 121ms/step - loss: 0.3517 - accuracy: 0.8601\n",
            "Epoch 6/75\n",
            "36/36 [==============================] - 7s 195ms/step - loss: 0.3416 - accuracy: 0.8640\n",
            "Epoch 7/75\n",
            "36/36 [==============================] - 5s 149ms/step - loss: 0.3089 - accuracy: 0.8702\n",
            "Epoch 8/75\n",
            "36/36 [==============================] - 6s 163ms/step - loss: 0.2688 - accuracy: 0.8853\n",
            "Epoch 9/75\n",
            "36/36 [==============================] - 5s 126ms/step - loss: 0.2391 - accuracy: 0.8915\n",
            "Epoch 10/75\n",
            "36/36 [==============================] - 4s 115ms/step - loss: 0.2752 - accuracy: 0.8782\n",
            "Epoch 11/75\n",
            "36/36 [==============================] - 6s 155ms/step - loss: 0.2057 - accuracy: 0.9128\n",
            "Epoch 12/75\n",
            "36/36 [==============================] - 4s 115ms/step - loss: 0.1856 - accuracy: 0.9194\n",
            "Epoch 13/75\n",
            "36/36 [==============================] - 4s 103ms/step - loss: 0.1834 - accuracy: 0.9190\n",
            "Epoch 14/75\n",
            "36/36 [==============================] - 5s 130ms/step - loss: 0.2003 - accuracy: 0.9110\n",
            "Epoch 15/75\n",
            "36/36 [==============================] - 4s 104ms/step - loss: 0.1653 - accuracy: 0.9238\n",
            "Epoch 16/75\n",
            "36/36 [==============================] - 4s 103ms/step - loss: 0.1552 - accuracy: 0.9371\n",
            "Epoch 17/75\n",
            "36/36 [==============================] - 5s 131ms/step - loss: 0.1477 - accuracy: 0.9358\n",
            "Epoch 18/75\n",
            "36/36 [==============================] - 4s 105ms/step - loss: 0.1356 - accuracy: 0.9460\n",
            "Epoch 19/75\n",
            "36/36 [==============================] - 4s 103ms/step - loss: 0.1396 - accuracy: 0.9420\n",
            "Epoch 20/75\n",
            "36/36 [==============================] - 4s 111ms/step - loss: 0.1269 - accuracy: 0.9517\n",
            "Epoch 21/75\n",
            "36/36 [==============================] - 4s 115ms/step - loss: 0.1156 - accuracy: 0.9553\n",
            "Epoch 22/75\n",
            "36/36 [==============================] - 3s 82ms/step - loss: 0.1012 - accuracy: 0.9624\n",
            "Epoch 23/75\n",
            "36/36 [==============================] - 2s 64ms/step - loss: 0.1088 - accuracy: 0.9539\n",
            "Epoch 24/75\n",
            "36/36 [==============================] - 2s 62ms/step - loss: 0.1178 - accuracy: 0.9544\n",
            "Epoch 25/75\n",
            "36/36 [==============================] - 4s 115ms/step - loss: 0.0858 - accuracy: 0.9708\n",
            "Epoch 26/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.0853 - accuracy: 0.9663\n",
            "Epoch 27/75\n",
            "36/36 [==============================] - 2s 59ms/step - loss: 0.0667 - accuracy: 0.9783\n",
            "Epoch 28/75\n",
            "36/36 [==============================] - 2s 59ms/step - loss: 0.0664 - accuracy: 0.9774\n",
            "Epoch 29/75\n",
            "36/36 [==============================] - 2s 59ms/step - loss: 0.0444 - accuracy: 0.9880\n",
            "Epoch 30/75\n",
            "36/36 [==============================] - 3s 91ms/step - loss: 0.0402 - accuracy: 0.9880\n",
            "Epoch 31/75\n",
            "36/36 [==============================] - 2s 66ms/step - loss: 0.0396 - accuracy: 0.9867\n",
            "Epoch 32/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.0305 - accuracy: 0.9920\n",
            "Epoch 33/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.0263 - accuracy: 0.9942\n",
            "Epoch 34/75\n",
            "36/36 [==============================] - 2s 60ms/step - loss: 0.0209 - accuracy: 0.9965\n",
            "Epoch 35/75\n",
            "36/36 [==============================] - 3s 75ms/step - loss: 0.0127 - accuracy: 0.9973\n",
            "Epoch 36/75\n",
            "36/36 [==============================] - 3s 81ms/step - loss: 0.0089 - accuracy: 1.0000\n",
            "Epoch 37/75\n",
            "36/36 [==============================] - 2s 60ms/step - loss: 0.0125 - accuracy: 0.9996\n",
            "Epoch 38/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.0090 - accuracy: 1.0000\n",
            "Epoch 39/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.0059 - accuracy: 0.9996\n",
            "Epoch 40/75\n",
            "36/36 [==============================] - 2s 65ms/step - loss: 0.0043 - accuracy: 1.0000\n",
            "Epoch 41/75\n",
            "36/36 [==============================] - 4s 114ms/step - loss: 0.0033 - accuracy: 1.0000\n",
            "Epoch 42/75\n",
            "36/36 [==============================] - 2s 66ms/step - loss: 0.0025 - accuracy: 1.0000\n",
            "Epoch 43/75\n",
            "36/36 [==============================] - 2s 62ms/step - loss: 0.0023 - accuracy: 1.0000\n",
            "Epoch 44/75\n",
            "36/36 [==============================] - 2s 60ms/step - loss: 0.0020 - accuracy: 1.0000\n",
            "Epoch 45/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.0016 - accuracy: 1.0000\n",
            "Epoch 46/75\n",
            "36/36 [==============================] - 4s 99ms/step - loss: 0.0015 - accuracy: 1.0000\n",
            "Epoch 47/75\n",
            "36/36 [==============================] - 2s 59ms/step - loss: 0.0013 - accuracy: 1.0000\n",
            "Epoch 48/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.0012 - accuracy: 1.0000\n",
            "Epoch 49/75\n",
            "36/36 [==============================] - 2s 60ms/step - loss: 0.0011 - accuracy: 1.0000\n",
            "Epoch 50/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 9.9483e-04 - accuracy: 1.0000\n",
            "Epoch 51/75\n",
            "36/36 [==============================] - 3s 84ms/step - loss: 9.2373e-04 - accuracy: 1.0000\n",
            "Epoch 52/75\n",
            "36/36 [==============================] - 3s 75ms/step - loss: 8.5188e-04 - accuracy: 1.0000\n",
            "Epoch 53/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 7.6477e-04 - accuracy: 1.0000\n",
            "Epoch 54/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 7.2304e-04 - accuracy: 1.0000\n",
            "Epoch 55/75\n",
            "36/36 [==============================] - 2s 59ms/step - loss: 6.6949e-04 - accuracy: 1.0000\n",
            "Epoch 56/75\n",
            "36/36 [==============================] - 2s 64ms/step - loss: 6.1046e-04 - accuracy: 1.0000\n",
            "Epoch 57/75\n",
            "36/36 [==============================] - 3s 92ms/step - loss: 5.8881e-04 - accuracy: 1.0000\n",
            "Epoch 58/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 5.4375e-04 - accuracy: 1.0000\n",
            "Epoch 59/75\n",
            "36/36 [==============================] - 2s 62ms/step - loss: 5.1393e-04 - accuracy: 1.0000\n",
            "Epoch 60/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 5.0470e-04 - accuracy: 1.0000\n",
            "Epoch 61/75\n",
            "36/36 [==============================] - 2s 62ms/step - loss: 4.4441e-04 - accuracy: 1.0000\n",
            "Epoch 62/75\n",
            "36/36 [==============================] - 3s 97ms/step - loss: 4.2609e-04 - accuracy: 1.0000\n",
            "Epoch 63/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 4.0342e-04 - accuracy: 1.0000\n",
            "Epoch 64/75\n",
            "36/36 [==============================] - 2s 60ms/step - loss: 3.7521e-04 - accuracy: 1.0000\n",
            "Epoch 65/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 3.6058e-04 - accuracy: 1.0000\n",
            "Epoch 66/75\n",
            "36/36 [==============================] - 2s 62ms/step - loss: 3.4242e-04 - accuracy: 1.0000\n",
            "Epoch 67/75\n",
            "36/36 [==============================] - 3s 83ms/step - loss: 3.1422e-04 - accuracy: 1.0000\n",
            "Epoch 68/75\n",
            "36/36 [==============================] - 3s 95ms/step - loss: 3.0181e-04 - accuracy: 1.0000\n",
            "Epoch 69/75\n",
            "36/36 [==============================] - 2s 65ms/step - loss: 2.8465e-04 - accuracy: 1.0000\n",
            "Epoch 70/75\n",
            "36/36 [==============================] - 2s 64ms/step - loss: 2.8490e-04 - accuracy: 1.0000\n",
            "Epoch 71/75\n",
            "36/36 [==============================] - 2s 65ms/step - loss: 2.6859e-04 - accuracy: 1.0000\n",
            "Epoch 72/75\n",
            "36/36 [==============================] - 3s 79ms/step - loss: 2.4409e-04 - accuracy: 1.0000\n",
            "Epoch 73/75\n",
            "36/36 [==============================] - 3s 77ms/step - loss: 2.4427e-04 - accuracy: 1.0000\n",
            "Epoch 74/75\n",
            "36/36 [==============================] - 2s 60ms/step - loss: 2.2629e-04 - accuracy: 1.0000\n",
            "Epoch 75/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 2.1539e-04 - accuracy: 1.0000\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7e1f843ee200>"
            ]
          },
          "metadata": {},
          "execution_count": 54
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = ann.predict(xtest)\n",
        "pred = (pred > 0.5)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ygF_Ps8D52Ch",
        "outputId": "041ff792-78df-4181-c115-05e0b0599f34"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "31/31 [==============================] - 1s 12ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(ytest, pred), precision_score(ytest, pred), recall_score(ytest, pred), f1_score(ytest, pred), cohen_kappa_score(ytest, pred), matthews_corrcoef(ytest, pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "GOQS1KaS55Qx",
        "outputId": "ee56daab-7a28-4bab-d617-7cc87ff8b003"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9618163054695562,\n",
              " 0.8870292887029289,\n",
              " 0.954954954954955,\n",
              " 0.9197396963123645,\n",
              " 0.8947335969911302,\n",
              " 0.8957819329109732)"
            ]
          },
          "metadata": {},
          "execution_count": 56
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(ytest, pred)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vEk7NsCy599k",
        "outputId": "fd3d5626-dfbe-495a-c104-8528ae9745c2"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.963855421686747"
            ]
          },
          "metadata": {},
          "execution_count": 57
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**ADASYN**"
      ],
      "metadata": {
        "id": "2vJgLs0gKo-P"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/LSA_TR.csv')"
      ],
      "metadata": {
        "id": "Rkf5DY9hKszx"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "lhhfUiUWK0YP"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.over_sampling import ADASYN\n",
        "ada = ADASYN()\n",
        "X, Y = ada.fit_resample(X, Y)"
      ],
      "metadata": {
        "id": "hFlHy5MdK2oP"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ],
      "metadata": {
        "id": "GK297FdwK4zM"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ],
      "metadata": {
        "id": "87qbiMHqK60c"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann = Sequential()"
      ],
      "metadata": {
        "id": "Ok4mV986K-MU"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Dense(256, activation = 'relu', input_shape=(X_train.shape[1], 1)))\n",
        "ann.add(Dense(256, activation = 'relu'))\n",
        "ann.add(Dense(128, activation = 'relu'))"
      ],
      "metadata": {
        "id": "mV0TXVInLA2G"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "ilRZtdzyLIj0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Flatten())"
      ],
      "metadata": {
        "id": "7mlZBczuLM-q"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Dense(64, activation='relu'))\n",
        "ann.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "YOVrwekbLPxH"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "XMUwCOclLS8v"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "f5OMIZ4SL0Bw",
        "outputId": "139b2dba-fdac-445d-d11a-8824af121306"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "66/66 [==============================] - 12s 122ms/step - loss: 0.5399 - accuracy: 0.7479\n",
            "Epoch 2/75\n",
            "66/66 [==============================] - 9s 135ms/step - loss: 0.4033 - accuracy: 0.8465\n",
            "Epoch 3/75\n",
            "66/66 [==============================] - 8s 115ms/step - loss: 0.3521 - accuracy: 0.8698\n",
            "Epoch 4/75\n",
            "66/66 [==============================] - 7s 105ms/step - loss: 0.2456 - accuracy: 0.9146\n",
            "Epoch 5/75\n",
            "66/66 [==============================] - 6s 95ms/step - loss: 0.1863 - accuracy: 0.9288\n",
            "Epoch 6/75\n",
            "66/66 [==============================] - 8s 122ms/step - loss: 0.1403 - accuracy: 0.9400\n",
            "Epoch 7/75\n",
            "66/66 [==============================] - 8s 120ms/step - loss: 0.1096 - accuracy: 0.9537\n",
            "Epoch 8/75\n",
            "66/66 [==============================] - 6s 91ms/step - loss: 0.0818 - accuracy: 0.9664\n",
            "Epoch 9/75\n",
            "66/66 [==============================] - 6s 95ms/step - loss: 0.0498 - accuracy: 0.9794\n",
            "Epoch 10/75\n",
            "66/66 [==============================] - 3s 52ms/step - loss: 0.0320 - accuracy: 0.9921\n",
            "Epoch 11/75\n",
            "66/66 [==============================] - 3s 52ms/step - loss: 0.0222 - accuracy: 0.9935\n",
            "Epoch 12/75\n",
            "66/66 [==============================] - 4s 57ms/step - loss: 0.0252 - accuracy: 0.9914\n",
            "Epoch 13/75\n",
            "66/66 [==============================] - 5s 69ms/step - loss: 0.0128 - accuracy: 0.9978\n",
            "Epoch 14/75\n",
            "66/66 [==============================] - 3s 53ms/step - loss: 0.0153 - accuracy: 0.9954\n",
            "Epoch 15/75\n",
            "66/66 [==============================] - 3s 51ms/step - loss: 0.0257 - accuracy: 0.9933\n",
            "Epoch 16/75\n",
            "66/66 [==============================] - 5s 71ms/step - loss: 0.0885 - accuracy: 0.9727\n",
            "Epoch 17/75\n",
            "66/66 [==============================] - 4s 61ms/step - loss: 0.0098 - accuracy: 0.9983\n",
            "Epoch 18/75\n",
            "66/66 [==============================] - 7s 110ms/step - loss: 0.0036 - accuracy: 1.0000\n",
            "Epoch 19/75\n",
            "66/66 [==============================] - 6s 89ms/step - loss: 0.0020 - accuracy: 1.0000\n",
            "Epoch 20/75\n",
            "66/66 [==============================] - 5s 82ms/step - loss: 0.0016 - accuracy: 1.0000\n",
            "Epoch 21/75\n",
            "66/66 [==============================] - 7s 100ms/step - loss: 0.0012 - accuracy: 1.0000\n",
            "Epoch 22/75\n",
            "66/66 [==============================] - 5s 83ms/step - loss: 0.0010 - accuracy: 1.0000\n",
            "Epoch 23/75\n",
            "66/66 [==============================] - 7s 109ms/step - loss: 8.2161e-04 - accuracy: 1.0000\n",
            "Epoch 24/75\n",
            "66/66 [==============================] - 6s 86ms/step - loss: 6.9245e-04 - accuracy: 1.0000\n",
            "Epoch 25/75\n",
            "66/66 [==============================] - 7s 109ms/step - loss: 6.5634e-04 - accuracy: 1.0000\n",
            "Epoch 26/75\n",
            "66/66 [==============================] - 6s 98ms/step - loss: 5.5271e-04 - accuracy: 1.0000\n",
            "Epoch 27/75\n",
            "66/66 [==============================] - 8s 121ms/step - loss: 4.5083e-04 - accuracy: 1.0000\n",
            "Epoch 28/75\n",
            "66/66 [==============================] - 7s 99ms/step - loss: 4.1728e-04 - accuracy: 1.0000\n",
            "Epoch 29/75\n",
            "66/66 [==============================] - 7s 113ms/step - loss: 3.6606e-04 - accuracy: 1.0000\n",
            "Epoch 30/75\n",
            "66/66 [==============================] - 8s 119ms/step - loss: 3.2182e-04 - accuracy: 1.0000\n",
            "Epoch 31/75\n",
            "66/66 [==============================] - 7s 106ms/step - loss: 2.8684e-04 - accuracy: 1.0000\n",
            "Epoch 32/75\n",
            "66/66 [==============================] - 8s 123ms/step - loss: 2.6070e-04 - accuracy: 1.0000\n",
            "Epoch 33/75\n",
            "66/66 [==============================] - 6s 94ms/step - loss: 2.3149e-04 - accuracy: 1.0000\n",
            "Epoch 34/75\n",
            "66/66 [==============================] - 7s 104ms/step - loss: 2.1302e-04 - accuracy: 1.0000\n",
            "Epoch 35/75\n",
            "66/66 [==============================] - 6s 91ms/step - loss: 1.9348e-04 - accuracy: 1.0000\n",
            "Epoch 36/75\n",
            "66/66 [==============================] - 7s 103ms/step - loss: 1.7941e-04 - accuracy: 1.0000\n",
            "Epoch 37/75\n",
            "66/66 [==============================] - 7s 101ms/step - loss: 1.6649e-04 - accuracy: 1.0000\n",
            "Epoch 38/75\n",
            "66/66 [==============================] - 8s 121ms/step - loss: 1.5381e-04 - accuracy: 1.0000\n",
            "Epoch 39/75\n",
            "66/66 [==============================] - 7s 108ms/step - loss: 1.4224e-04 - accuracy: 1.0000\n",
            "Epoch 40/75\n",
            "66/66 [==============================] - 8s 117ms/step - loss: 1.3070e-04 - accuracy: 1.0000\n",
            "Epoch 41/75\n",
            "66/66 [==============================] - 8s 122ms/step - loss: 1.2166e-04 - accuracy: 1.0000\n",
            "Epoch 42/75\n",
            "66/66 [==============================] - 7s 104ms/step - loss: 1.1294e-04 - accuracy: 1.0000\n",
            "Epoch 43/75\n",
            "66/66 [==============================] - 8s 129ms/step - loss: 1.0735e-04 - accuracy: 1.0000\n",
            "Epoch 44/75\n",
            "66/66 [==============================] - 7s 100ms/step - loss: 1.0099e-04 - accuracy: 1.0000\n",
            "Epoch 45/75\n",
            "66/66 [==============================] - 7s 104ms/step - loss: 9.1815e-05 - accuracy: 1.0000\n",
            "Epoch 46/75\n",
            "66/66 [==============================] - 6s 92ms/step - loss: 8.6771e-05 - accuracy: 1.0000\n",
            "Epoch 47/75\n",
            "66/66 [==============================] - 7s 113ms/step - loss: 8.1428e-05 - accuracy: 1.0000\n",
            "Epoch 48/75\n",
            "66/66 [==============================] - 6s 91ms/step - loss: 7.6056e-05 - accuracy: 1.0000\n",
            "Epoch 49/75\n",
            "66/66 [==============================] - 7s 100ms/step - loss: 7.1028e-05 - accuracy: 1.0000\n",
            "Epoch 50/75\n",
            "66/66 [==============================] - 7s 101ms/step - loss: 6.7708e-05 - accuracy: 1.0000\n",
            "Epoch 51/75\n",
            "66/66 [==============================] - 8s 122ms/step - loss: 6.3603e-05 - accuracy: 1.0000\n",
            "Epoch 52/75\n",
            "66/66 [==============================] - 6s 87ms/step - loss: 6.0516e-05 - accuracy: 1.0000\n",
            "Epoch 53/75\n",
            "66/66 [==============================] - 7s 109ms/step - loss: 5.6593e-05 - accuracy: 1.0000\n",
            "Epoch 54/75\n",
            "66/66 [==============================] - 6s 98ms/step - loss: 5.4048e-05 - accuracy: 1.0000\n",
            "Epoch 55/75\n",
            "66/66 [==============================] - 7s 107ms/step - loss: 5.0447e-05 - accuracy: 1.0000\n",
            "Epoch 56/75\n",
            "66/66 [==============================] - 6s 91ms/step - loss: 4.7858e-05 - accuracy: 1.0000\n",
            "Epoch 57/75\n",
            "66/66 [==============================] - 7s 100ms/step - loss: 4.5128e-05 - accuracy: 1.0000\n",
            "Epoch 58/75\n",
            "66/66 [==============================] - 7s 112ms/step - loss: 4.3392e-05 - accuracy: 1.0000\n",
            "Epoch 59/75\n",
            "66/66 [==============================] - 8s 127ms/step - loss: 4.0665e-05 - accuracy: 1.0000\n",
            "Epoch 60/75\n",
            "66/66 [==============================] - 7s 113ms/step - loss: 3.8656e-05 - accuracy: 1.0000\n",
            "Epoch 61/75\n",
            "66/66 [==============================] - 6s 96ms/step - loss: 3.6438e-05 - accuracy: 1.0000\n",
            "Epoch 62/75\n",
            "66/66 [==============================] - 6s 92ms/step - loss: 3.4908e-05 - accuracy: 1.0000\n",
            "Epoch 63/75\n",
            "66/66 [==============================] - 4s 61ms/step - loss: 3.5167e-05 - accuracy: 1.0000\n",
            "Epoch 64/75\n",
            "66/66 [==============================] - 4s 54ms/step - loss: 3.1085e-05 - accuracy: 1.0000\n",
            "Epoch 65/75\n",
            "66/66 [==============================] - 4s 53ms/step - loss: 2.9505e-05 - accuracy: 1.0000\n",
            "Epoch 66/75\n",
            "66/66 [==============================] - 5s 72ms/step - loss: 2.8188e-05 - accuracy: 1.0000\n",
            "Epoch 67/75\n",
            "66/66 [==============================] - 4s 54ms/step - loss: 2.6857e-05 - accuracy: 1.0000\n",
            "Epoch 68/75\n",
            "66/66 [==============================] - 4s 54ms/step - loss: 2.5590e-05 - accuracy: 1.0000\n",
            "Epoch 69/75\n",
            "66/66 [==============================] - 5s 72ms/step - loss: 2.4268e-05 - accuracy: 1.0000\n",
            "Epoch 70/75\n",
            "66/66 [==============================] - 4s 54ms/step - loss: 2.2849e-05 - accuracy: 1.0000\n",
            "Epoch 71/75\n",
            "66/66 [==============================] - 4s 53ms/step - loss: 2.2011e-05 - accuracy: 1.0000\n",
            "Epoch 72/75\n",
            "66/66 [==============================] - 4s 68ms/step - loss: 2.0860e-05 - accuracy: 1.0000\n",
            "Epoch 73/75\n",
            "66/66 [==============================] - 4s 56ms/step - loss: 1.9970e-05 - accuracy: 1.0000\n",
            "Epoch 74/75\n",
            "66/66 [==============================] - 4s 54ms/step - loss: 1.9096e-05 - accuracy: 1.0000\n",
            "Epoch 75/75\n",
            "66/66 [==============================] - 5s 76ms/step - loss: 1.8085e-05 - accuracy: 1.0000\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x794cc0698ee0>"
            ]
          },
          "metadata": {},
          "execution_count": 114
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = ann.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "h0QqJqcvLWPn",
        "outputId": "c3b7c18f-a735-4b49-a2d4-8258fd0e7c91"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "33/33 [==============================] - 1s 16ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "jtn8S-mmLY2x",
        "outputId": "8c04bad8-56ec-437d-a1a5-cef4ca77523c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9923224568138196,\n",
              " 1.0,\n",
              " 0.9853479853479854,\n",
              " 0.9846208341942911,\n",
              " 0.984737295493632)"
            ]
          },
          "metadata": {},
          "execution_count": 116
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ity9sfSxLcEm",
        "outputId": "a1f3f8a8-a68e-4281-92f9-71b99b92dbdc"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9841269841269841"
            ]
          },
          "metadata": {},
          "execution_count": 117
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**SMOTETomek**"
      ],
      "metadata": {
        "id": "01jT9gK7Le4u"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/LSA_TR.csv')"
      ],
      "metadata": {
        "id": "VvghL9npLhTI"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "8rBz0oxcLlef"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.combine import SMOTETomek\n",
        "smt = SMOTETomek()\n",
        "X, Y = smt.fit_resample(X, Y)"
      ],
      "metadata": {
        "id": "2GxdhSlAL5bA"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ],
      "metadata": {
        "id": "Y7lKje5xL7g3"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ],
      "metadata": {
        "id": "7QoNx5ILL9vo"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann = Sequential()"
      ],
      "metadata": {
        "id": "FlVqsR0NL_oq"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Dense(256, activation = 'relu', input_shape=(X_train.shape[1], 1)))\n",
        "ann.add(Dense(256, activation = 'relu'))\n",
        "ann.add(Dense(128, activation = 'relu'))"
      ],
      "metadata": {
        "id": "FFN-lq1dMCvQ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "6ZkNfWKIMP_O"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Flatten())"
      ],
      "metadata": {
        "id": "-ikam_meMTqu"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Dense(64, activation='relu'))\n",
        "ann.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "3WcWpWIjMWTG"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "F5fMR9dAMY_O"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8K5phFqkMb2W",
        "outputId": "cb45e68a-ce66-407d-dd80-da42084ef5ba"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "65/65 [==============================] - 5s 54ms/step - loss: 0.5680 - accuracy: 0.7060\n",
            "Epoch 2/75\n",
            "65/65 [==============================] - 4s 54ms/step - loss: 0.3893 - accuracy: 0.8487\n",
            "Epoch 3/75\n",
            "65/65 [==============================] - 5s 72ms/step - loss: 0.2773 - accuracy: 0.9013\n",
            "Epoch 4/75\n",
            "65/65 [==============================] - 3s 54ms/step - loss: 0.2400 - accuracy: 0.9167\n",
            "Epoch 5/75\n",
            "65/65 [==============================] - 4s 54ms/step - loss: 0.2023 - accuracy: 0.9288\n",
            "Epoch 6/75\n",
            "65/65 [==============================] - 5s 74ms/step - loss: 0.1584 - accuracy: 0.9485\n",
            "Epoch 7/75\n",
            "65/65 [==============================] - 4s 54ms/step - loss: 0.1331 - accuracy: 0.9574\n",
            "Epoch 8/75\n",
            "65/65 [==============================] - 7s 103ms/step - loss: 0.1203 - accuracy: 0.9622\n",
            "Epoch 9/75\n",
            "65/65 [==============================] - 7s 104ms/step - loss: 0.0969 - accuracy: 0.9702\n",
            "Epoch 10/75\n",
            "65/65 [==============================] - 4s 59ms/step - loss: 0.0750 - accuracy: 0.9758\n",
            "Epoch 11/75\n",
            "65/65 [==============================] - 5s 73ms/step - loss: 0.0591 - accuracy: 0.9828\n",
            "Epoch 12/75\n",
            "65/65 [==============================] - 4s 55ms/step - loss: 0.0600 - accuracy: 0.9804\n",
            "Epoch 13/75\n",
            "65/65 [==============================] - 4s 55ms/step - loss: 0.0445 - accuracy: 0.9867\n",
            "Epoch 14/75\n",
            "65/65 [==============================] - 5s 70ms/step - loss: 0.0518 - accuracy: 0.9838\n",
            "Epoch 15/75\n",
            "65/65 [==============================] - 4s 56ms/step - loss: 0.0223 - accuracy: 0.9942\n",
            "Epoch 16/75\n",
            "65/65 [==============================] - 4s 54ms/step - loss: 0.0193 - accuracy: 0.9947\n",
            "Epoch 17/75\n",
            "65/65 [==============================] - 4s 57ms/step - loss: 0.0224 - accuracy: 0.9935\n",
            "Epoch 18/75\n",
            "65/65 [==============================] - 5s 70ms/step - loss: 0.0100 - accuracy: 0.9976\n",
            "Epoch 19/75\n",
            "65/65 [==============================] - 4s 55ms/step - loss: 0.0114 - accuracy: 0.9966\n",
            "Epoch 20/75\n",
            "65/65 [==============================] - 4s 55ms/step - loss: 0.0103 - accuracy: 0.9976\n",
            "Epoch 21/75\n",
            "65/65 [==============================] - 5s 73ms/step - loss: 0.0048 - accuracy: 0.9990\n",
            "Epoch 22/75\n",
            "65/65 [==============================] - 4s 54ms/step - loss: 0.0024 - accuracy: 0.9998\n",
            "Epoch 23/75\n",
            "65/65 [==============================] - 3s 53ms/step - loss: 0.0014 - accuracy: 1.0000\n",
            "Epoch 24/75\n",
            "65/65 [==============================] - 6s 85ms/step - loss: 0.0012 - accuracy: 1.0000\n",
            "Epoch 25/75\n",
            "65/65 [==============================] - 6s 96ms/step - loss: 9.0405e-04 - accuracy: 1.0000\n",
            "Epoch 26/75\n",
            "65/65 [==============================] - 7s 115ms/step - loss: 7.6362e-04 - accuracy: 1.0000\n",
            "Epoch 27/75\n",
            "65/65 [==============================] - 4s 59ms/step - loss: 6.7143e-04 - accuracy: 1.0000\n",
            "Epoch 28/75\n",
            "65/65 [==============================] - 3s 54ms/step - loss: 6.1378e-04 - accuracy: 1.0000\n",
            "Epoch 29/75\n",
            "65/65 [==============================] - 5s 73ms/step - loss: 5.3689e-04 - accuracy: 1.0000\n",
            "Epoch 30/75\n",
            "65/65 [==============================] - 6s 95ms/step - loss: 4.9624e-04 - accuracy: 1.0000\n",
            "Epoch 31/75\n",
            "65/65 [==============================] - 6s 95ms/step - loss: 4.7526e-04 - accuracy: 1.0000\n",
            "Epoch 32/75\n",
            "65/65 [==============================] - 5s 71ms/step - loss: 3.9678e-04 - accuracy: 1.0000\n",
            "Epoch 33/75\n",
            "65/65 [==============================] - 6s 91ms/step - loss: 3.5901e-04 - accuracy: 1.0000\n",
            "Epoch 34/75\n",
            "65/65 [==============================] - 4s 68ms/step - loss: 3.2147e-04 - accuracy: 1.0000\n",
            "Epoch 35/75\n",
            "65/65 [==============================] - 4s 55ms/step - loss: 2.9496e-04 - accuracy: 1.0000\n",
            "Epoch 36/75\n",
            "65/65 [==============================] - 4s 55ms/step - loss: 2.6994e-04 - accuracy: 1.0000\n",
            "Epoch 37/75\n",
            "65/65 [==============================] - 5s 73ms/step - loss: 2.5375e-04 - accuracy: 1.0000\n",
            "Epoch 38/75\n",
            "65/65 [==============================] - 3s 53ms/step - loss: 2.3448e-04 - accuracy: 1.0000\n",
            "Epoch 39/75\n",
            "65/65 [==============================] - 4s 55ms/step - loss: 2.1880e-04 - accuracy: 1.0000\n",
            "Epoch 40/75\n",
            "65/65 [==============================] - 5s 74ms/step - loss: 2.0146e-04 - accuracy: 1.0000\n",
            "Epoch 41/75\n",
            "65/65 [==============================] - 3s 53ms/step - loss: 1.8899e-04 - accuracy: 1.0000\n",
            "Epoch 42/75\n",
            "65/65 [==============================] - 3s 53ms/step - loss: 1.8345e-04 - accuracy: 1.0000\n",
            "Epoch 43/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 1.6296e-04 - accuracy: 1.0000\n",
            "Epoch 44/75\n",
            "65/65 [==============================] - 4s 61ms/step - loss: 1.5580e-04 - accuracy: 1.0000\n",
            "Epoch 45/75\n",
            "65/65 [==============================] - 3s 53ms/step - loss: 1.4246e-04 - accuracy: 1.0000\n",
            "Epoch 46/75\n",
            "65/65 [==============================] - 3s 53ms/step - loss: 1.3925e-04 - accuracy: 1.0000\n",
            "Epoch 47/75\n",
            "65/65 [==============================] - 5s 73ms/step - loss: 1.2914e-04 - accuracy: 1.0000\n",
            "Epoch 48/75\n",
            "65/65 [==============================] - 4s 55ms/step - loss: 1.1904e-04 - accuracy: 1.0000\n",
            "Epoch 49/75\n",
            "65/65 [==============================] - 3s 54ms/step - loss: 1.1101e-04 - accuracy: 1.0000\n",
            "Epoch 50/75\n",
            "65/65 [==============================] - 5s 74ms/step - loss: 1.0385e-04 - accuracy: 1.0000\n",
            "Epoch 51/75\n",
            "65/65 [==============================] - 4s 54ms/step - loss: 1.0023e-04 - accuracy: 1.0000\n",
            "Epoch 52/75\n",
            "65/65 [==============================] - 4s 54ms/step - loss: 9.2891e-05 - accuracy: 1.0000\n",
            "Epoch 53/75\n",
            "65/65 [==============================] - 5s 71ms/step - loss: 8.8017e-05 - accuracy: 1.0000\n",
            "Epoch 54/75\n",
            "65/65 [==============================] - 5s 82ms/step - loss: 8.3139e-05 - accuracy: 1.0000\n",
            "Epoch 55/75\n",
            "65/65 [==============================] - 5s 73ms/step - loss: 7.8437e-05 - accuracy: 1.0000\n",
            "Epoch 56/75\n",
            "65/65 [==============================] - 5s 74ms/step - loss: 7.3945e-05 - accuracy: 1.0000\n",
            "Epoch 57/75\n",
            "65/65 [==============================] - 3s 53ms/step - loss: 6.8636e-05 - accuracy: 1.0000\n",
            "Epoch 58/75\n",
            "65/65 [==============================] - 3s 54ms/step - loss: 6.7378e-05 - accuracy: 1.0000\n",
            "Epoch 59/75\n",
            "65/65 [==============================] - 5s 74ms/step - loss: 6.3596e-05 - accuracy: 1.0000\n",
            "Epoch 60/75\n",
            "65/65 [==============================] - 3s 53ms/step - loss: 6.0688e-05 - accuracy: 1.0000\n",
            "Epoch 61/75\n",
            "65/65 [==============================] - 4s 55ms/step - loss: 5.6261e-05 - accuracy: 1.0000\n",
            "Epoch 62/75\n",
            "65/65 [==============================] - 5s 70ms/step - loss: 5.3544e-05 - accuracy: 1.0000\n",
            "Epoch 63/75\n",
            "65/65 [==============================] - 4s 60ms/step - loss: 5.0735e-05 - accuracy: 1.0000\n",
            "Epoch 64/75\n",
            "65/65 [==============================] - 3s 54ms/step - loss: 4.8335e-05 - accuracy: 1.0000\n",
            "Epoch 65/75\n",
            "65/65 [==============================] - 4s 60ms/step - loss: 4.6235e-05 - accuracy: 1.0000\n",
            "Epoch 66/75\n",
            "65/65 [==============================] - 4s 68ms/step - loss: 4.3394e-05 - accuracy: 1.0000\n",
            "Epoch 67/75\n",
            "65/65 [==============================] - 4s 55ms/step - loss: 4.1042e-05 - accuracy: 1.0000\n",
            "Epoch 68/75\n",
            "65/65 [==============================] - 5s 84ms/step - loss: 4.0714e-05 - accuracy: 1.0000\n",
            "Epoch 69/75\n",
            "65/65 [==============================] - 6s 90ms/step - loss: 3.7851e-05 - accuracy: 1.0000\n",
            "Epoch 70/75\n",
            "65/65 [==============================] - 3s 54ms/step - loss: 3.5782e-05 - accuracy: 1.0000\n",
            "Epoch 71/75\n",
            "65/65 [==============================] - 4s 68ms/step - loss: 3.4324e-05 - accuracy: 1.0000\n",
            "Epoch 72/75\n",
            "65/65 [==============================] - 4s 58ms/step - loss: 3.2557e-05 - accuracy: 1.0000\n",
            "Epoch 73/75\n",
            "65/65 [==============================] - 4s 54ms/step - loss: 3.1351e-05 - accuracy: 1.0000\n",
            "Epoch 74/75\n",
            "65/65 [==============================] - 4s 56ms/step - loss: 2.9609e-05 - accuracy: 1.0000\n",
            "Epoch 75/75\n",
            "65/65 [==============================] - 5s 72ms/step - loss: 2.8259e-05 - accuracy: 1.0000\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x794cc034a320>"
            ]
          },
          "metadata": {},
          "execution_count": 129
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = ann.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-2dix5yWMe4n",
        "outputId": "d4341007-b21d-4bc2-cd77-00f6c1d46064"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "33/33 [==============================] - 0s 10ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "WqF_R7LYMhpO",
        "outputId": "dcd818de-3bfc-48c6-ccfd-4a47e5490d7b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9874031007751938,\n",
              " 0.9980353634577603,\n",
              " 0.9769230769230769,\n",
              " 0.9748088507007556,\n",
              " 0.975030380483132)"
            ]
          },
          "metadata": {},
          "execution_count": 131
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Z_yC0dwnMk9f",
        "outputId": "1d2f185a-137e-4844-c7c5-4cffd2099e84"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9770554493307839"
            ]
          },
          "metadata": {},
          "execution_count": 132
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**NearMiss**"
      ],
      "metadata": {
        "id": "b3DMzRfSMpAa"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/LSA_TR.csv')"
      ],
      "metadata": {
        "id": "5v8p7-knMrjO"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "au4fN1TdMxMu"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.under_sampling import NearMiss\n",
        "nm = NearMiss()\n",
        "X, Y = nm.fit_resample(X, Y)"
      ],
      "metadata": {
        "id": "nkpFl_HXMzAC"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ],
      "metadata": {
        "id": "bq1lTBa_M0Z4"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ],
      "metadata": {
        "id": "jsDD3dPfM2k4"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann = Sequential()"
      ],
      "metadata": {
        "id": "_xAtNkchM4mo"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Dense(256, activation = 'relu', input_shape=(X_train.shape[1], 1)))\n",
        "ann.add(Dense(256, activation = 'relu'))\n",
        "ann.add(Dense(128, activation = 'relu'))"
      ],
      "metadata": {
        "id": "HD2lRfV_M7uN"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "EjJSPF2DM_QJ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Flatten())"
      ],
      "metadata": {
        "id": "A8c5m3PfNCNP"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Dense(64, activation='relu'))\n",
        "ann.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "XelManEcNEuX"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "O4iKQ4OgNHTQ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "CJqeYem3NJ9-",
        "outputId": "495705ac-289d-4b7a-ec1d-84ec4045de00"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "17/17 [==============================] - 3s 96ms/step - loss: 0.6716 - accuracy: 0.5736\n",
            "Epoch 2/75\n",
            "17/17 [==============================] - 1s 76ms/step - loss: 0.5405 - accuracy: 0.7771\n",
            "Epoch 3/75\n",
            "17/17 [==============================] - 1s 52ms/step - loss: 0.3835 - accuracy: 0.8769\n",
            "Epoch 4/75\n",
            "17/17 [==============================] - 1s 53ms/step - loss: 0.3174 - accuracy: 0.8924\n",
            "Epoch 5/75\n",
            "17/17 [==============================] - 1s 53ms/step - loss: 0.2687 - accuracy: 0.8924\n",
            "Epoch 6/75\n",
            "17/17 [==============================] - 1s 49ms/step - loss: 0.1953 - accuracy: 0.9157\n",
            "Epoch 7/75\n",
            "17/17 [==============================] - 1s 53ms/step - loss: 0.1798 - accuracy: 0.9312\n",
            "Epoch 8/75\n",
            "17/17 [==============================] - 1s 52ms/step - loss: 0.1448 - accuracy: 0.9399\n",
            "Epoch 9/75\n",
            "17/17 [==============================] - 1s 52ms/step - loss: 0.0968 - accuracy: 0.9690\n",
            "Epoch 10/75\n",
            "17/17 [==============================] - 1s 51ms/step - loss: 0.0771 - accuracy: 0.9767\n",
            "Epoch 11/75\n",
            "17/17 [==============================] - 1s 52ms/step - loss: 0.0736 - accuracy: 0.9797\n",
            "Epoch 12/75\n",
            "17/17 [==============================] - 1s 51ms/step - loss: 0.0894 - accuracy: 0.9700\n",
            "Epoch 13/75\n",
            "17/17 [==============================] - 1s 57ms/step - loss: 0.0495 - accuracy: 0.9893\n",
            "Epoch 14/75\n",
            "17/17 [==============================] - 2s 92ms/step - loss: 0.0420 - accuracy: 0.9952\n",
            "Epoch 15/75\n",
            "17/17 [==============================] - 1s 79ms/step - loss: 0.0458 - accuracy: 0.9864\n",
            "Epoch 16/75\n",
            "17/17 [==============================] - 1s 50ms/step - loss: 0.0662 - accuracy: 0.9777\n",
            "Epoch 17/75\n",
            "17/17 [==============================] - 1s 50ms/step - loss: 0.0308 - accuracy: 0.9942\n",
            "Epoch 18/75\n",
            "17/17 [==============================] - 1s 51ms/step - loss: 0.0264 - accuracy: 0.9952\n",
            "Epoch 19/75\n",
            "17/17 [==============================] - 1s 51ms/step - loss: 0.0298 - accuracy: 0.9922\n",
            "Epoch 20/75\n",
            "17/17 [==============================] - 1s 52ms/step - loss: 0.0309 - accuracy: 0.9932\n",
            "Epoch 21/75\n",
            "17/17 [==============================] - 1s 52ms/step - loss: 0.0393 - accuracy: 0.9884\n",
            "Epoch 22/75\n",
            "17/17 [==============================] - 1s 51ms/step - loss: 0.0400 - accuracy: 0.9826\n",
            "Epoch 23/75\n",
            "17/17 [==============================] - 1s 53ms/step - loss: 0.0532 - accuracy: 0.9816\n",
            "Epoch 24/75\n",
            "17/17 [==============================] - 1s 52ms/step - loss: 0.0340 - accuracy: 0.9932\n",
            "Epoch 25/75\n",
            "17/17 [==============================] - 1s 55ms/step - loss: 0.0240 - accuracy: 0.9913\n",
            "Epoch 26/75\n",
            "17/17 [==============================] - 1s 52ms/step - loss: 0.0436 - accuracy: 0.9884\n",
            "Epoch 27/75\n",
            "17/17 [==============================] - 2s 94ms/step - loss: 0.0267 - accuracy: 0.9942\n",
            "Epoch 28/75\n",
            "17/17 [==============================] - 1s 82ms/step - loss: 0.0356 - accuracy: 0.9893\n",
            "Epoch 29/75\n",
            "17/17 [==============================] - 1s 53ms/step - loss: 0.0301 - accuracy: 0.9893\n",
            "Epoch 30/75\n",
            "17/17 [==============================] - 1s 52ms/step - loss: 0.0166 - accuracy: 0.9961\n",
            "Epoch 31/75\n",
            "17/17 [==============================] - 1s 52ms/step - loss: 0.0083 - accuracy: 1.0000\n",
            "Epoch 32/75\n",
            "17/17 [==============================] - 1s 52ms/step - loss: 0.0100 - accuracy: 0.9990\n",
            "Epoch 33/75\n",
            "17/17 [==============================] - 1s 52ms/step - loss: 0.0089 - accuracy: 0.9981\n",
            "Epoch 34/75\n",
            "17/17 [==============================] - 1s 51ms/step - loss: 0.0082 - accuracy: 0.9990\n",
            "Epoch 35/75\n",
            "17/17 [==============================] - 1s 50ms/step - loss: 0.0127 - accuracy: 0.9961\n",
            "Epoch 36/75\n",
            "17/17 [==============================] - 1s 52ms/step - loss: 0.0081 - accuracy: 0.9990\n",
            "Epoch 37/75\n",
            "17/17 [==============================] - 1s 51ms/step - loss: 0.0069 - accuracy: 0.9990\n",
            "Epoch 38/75\n",
            "17/17 [==============================] - 1s 51ms/step - loss: 0.0072 - accuracy: 0.9981\n",
            "Epoch 39/75\n",
            "17/17 [==============================] - 1s 52ms/step - loss: 0.0047 - accuracy: 1.0000\n",
            "Epoch 40/75\n",
            "17/17 [==============================] - 2s 94ms/step - loss: 0.0035 - accuracy: 1.0000\n",
            "Epoch 41/75\n",
            "17/17 [==============================] - 1s 84ms/step - loss: 0.0036 - accuracy: 1.0000\n",
            "Epoch 42/75\n",
            "17/17 [==============================] - 1s 52ms/step - loss: 0.0029 - accuracy: 1.0000\n",
            "Epoch 43/75\n",
            "17/17 [==============================] - 1s 51ms/step - loss: 0.0026 - accuracy: 1.0000\n",
            "Epoch 44/75\n",
            "17/17 [==============================] - 1s 54ms/step - loss: 0.0025 - accuracy: 1.0000\n",
            "Epoch 45/75\n",
            "17/17 [==============================] - 1s 51ms/step - loss: 0.0024 - accuracy: 1.0000\n",
            "Epoch 46/75\n",
            "17/17 [==============================] - 1s 52ms/step - loss: 0.0026 - accuracy: 1.0000\n",
            "Epoch 47/75\n",
            "17/17 [==============================] - 1s 54ms/step - loss: 0.0027 - accuracy: 1.0000\n",
            "Epoch 48/75\n",
            "17/17 [==============================] - 1s 52ms/step - loss: 0.0020 - accuracy: 1.0000\n",
            "Epoch 49/75\n",
            "17/17 [==============================] - 1s 52ms/step - loss: 0.0018 - accuracy: 1.0000\n",
            "Epoch 50/75\n",
            "17/17 [==============================] - 1s 51ms/step - loss: 0.0019 - accuracy: 1.0000\n",
            "Epoch 51/75\n",
            "17/17 [==============================] - 1s 55ms/step - loss: 0.0017 - accuracy: 1.0000\n",
            "Epoch 52/75\n",
            "17/17 [==============================] - 1s 62ms/step - loss: 0.0017 - accuracy: 1.0000\n",
            "Epoch 53/75\n",
            "17/17 [==============================] - 2s 98ms/step - loss: 0.0017 - accuracy: 1.0000\n",
            "Epoch 54/75\n",
            "17/17 [==============================] - 1s 75ms/step - loss: 0.0014 - accuracy: 1.0000\n",
            "Epoch 55/75\n",
            "17/17 [==============================] - 1s 55ms/step - loss: 0.0014 - accuracy: 1.0000\n",
            "Epoch 56/75\n",
            "17/17 [==============================] - 1s 56ms/step - loss: 0.0012 - accuracy: 1.0000\n",
            "Epoch 57/75\n",
            "17/17 [==============================] - 1s 57ms/step - loss: 0.0011 - accuracy: 1.0000\n",
            "Epoch 58/75\n",
            "17/17 [==============================] - 1s 54ms/step - loss: 0.0011 - accuracy: 1.0000\n",
            "Epoch 59/75\n",
            "17/17 [==============================] - 1s 54ms/step - loss: 0.0010 - accuracy: 1.0000\n",
            "Epoch 60/75\n",
            "17/17 [==============================] - 1s 56ms/step - loss: 9.8726e-04 - accuracy: 1.0000\n",
            "Epoch 61/75\n",
            "17/17 [==============================] - 1s 53ms/step - loss: 0.0010 - accuracy: 1.0000\n",
            "Epoch 62/75\n",
            "17/17 [==============================] - 1s 52ms/step - loss: 0.0010 - accuracy: 1.0000\n",
            "Epoch 63/75\n",
            "17/17 [==============================] - 1s 53ms/step - loss: 8.3854e-04 - accuracy: 1.0000\n",
            "Epoch 64/75\n",
            "17/17 [==============================] - 1s 60ms/step - loss: 9.7921e-04 - accuracy: 1.0000\n",
            "Epoch 65/75\n",
            "17/17 [==============================] - 2s 121ms/step - loss: 8.5186e-04 - accuracy: 1.0000\n",
            "Epoch 66/75\n",
            "17/17 [==============================] - 2s 106ms/step - loss: 8.3410e-04 - accuracy: 1.0000\n",
            "Epoch 67/75\n",
            "17/17 [==============================] - 2s 89ms/step - loss: 7.3130e-04 - accuracy: 1.0000\n",
            "Epoch 68/75\n",
            "17/17 [==============================] - 2s 89ms/step - loss: 7.2396e-04 - accuracy: 1.0000\n",
            "Epoch 69/75\n",
            "17/17 [==============================] - 2s 92ms/step - loss: 7.4954e-04 - accuracy: 1.0000\n",
            "Epoch 70/75\n",
            "17/17 [==============================] - 1s 76ms/step - loss: 7.0696e-04 - accuracy: 1.0000\n",
            "Epoch 71/75\n",
            "17/17 [==============================] - 1s 55ms/step - loss: 6.5516e-04 - accuracy: 1.0000\n",
            "Epoch 72/75\n",
            "17/17 [==============================] - 1s 55ms/step - loss: 6.5547e-04 - accuracy: 1.0000\n",
            "Epoch 73/75\n",
            "17/17 [==============================] - 1s 55ms/step - loss: 6.7499e-04 - accuracy: 1.0000\n",
            "Epoch 74/75\n",
            "17/17 [==============================] - 1s 71ms/step - loss: 6.8202e-04 - accuracy: 1.0000\n",
            "Epoch 75/75\n",
            "17/17 [==============================] - 2s 97ms/step - loss: 6.2322e-04 - accuracy: 1.0000\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x794cc01b2920>"
            ]
          },
          "metadata": {},
          "execution_count": 144
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = ann.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "PVAPpM6ONMce",
        "outputId": "b84c59fb-8f67-446a-e6af-03eadd728732"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "9/9 [==============================] - 0s 16ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "t57DeeCoNO7A",
        "outputId": "9b4b6792-8b37-4aee-bcce-b2f11a7856ff"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9844961240310077,\n",
              " 1.0,\n",
              " 0.9701492537313433,\n",
              " 0.968982928588603,\n",
              " 0.96944937441428)"
            ]
          },
          "metadata": {},
          "execution_count": 146
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "PBYa3tmaNSgm",
        "outputId": "87cedf2d-d532-4628-e493-ccdb15ddcb18"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.96875"
            ]
          },
          "metadata": {},
          "execution_count": 147
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **ANN(NMBroto)**"
      ],
      "metadata": {
        "id": "RzCkZcczQedc"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Imbalanced**"
      ],
      "metadata": {
        "id": "3X35mDFRQqaV"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/NMB-TR.csv')"
      ],
      "metadata": {
        "id": "qRkXDuicQigF"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "LRhjCP53QtR0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ],
      "metadata": {
        "id": "oEieiG22QyY8"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ],
      "metadata": {
        "id": "OKBcn7CtQ0lU"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann = Sequential()"
      ],
      "metadata": {
        "id": "cDY5dgsdQ3OM"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Dense(256, activation = 'relu', input_shape=(X_train.shape[1], 1)))\n",
        "ann.add(Dense(256, activation = 'relu'))\n",
        "ann.add(Dense(128, activation = 'relu'))"
      ],
      "metadata": {
        "id": "Cejx8lzZQ5bV"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "gHQA0j_aQ7el"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Flatten())"
      ],
      "metadata": {
        "id": "Xps5lgb8Q9mc"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Dense(64, activation='relu'))\n",
        "ann.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "ciWzlDYcQ_vE"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "ftmFj9l6RCy0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "azHA139lRFMN",
        "outputId": "6d07ca76-2689-4670-80fc-fe5193c8b2a4"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "41/41 [==============================] - 4s 33ms/step - loss: 0.5258 - accuracy: 0.8013\n",
            "Epoch 2/75\n",
            "41/41 [==============================] - 1s 31ms/step - loss: 0.4854 - accuracy: 0.8021\n",
            "Epoch 3/75\n",
            "41/41 [==============================] - 1s 33ms/step - loss: 0.4579 - accuracy: 0.8021\n",
            "Epoch 4/75\n",
            "41/41 [==============================] - 1s 33ms/step - loss: 0.4530 - accuracy: 0.8021\n",
            "Epoch 5/75\n",
            "41/41 [==============================] - 1s 32ms/step - loss: 0.4325 - accuracy: 0.8021\n",
            "Epoch 6/75\n",
            "41/41 [==============================] - 2s 38ms/step - loss: 0.4281 - accuracy: 0.8025\n",
            "Epoch 7/75\n",
            "41/41 [==============================] - 1s 34ms/step - loss: 0.4207 - accuracy: 0.8033\n",
            "Epoch 8/75\n",
            "41/41 [==============================] - 2s 48ms/step - loss: 0.4097 - accuracy: 0.8129\n",
            "Epoch 9/75\n",
            "41/41 [==============================] - 2s 36ms/step - loss: 0.3984 - accuracy: 0.8125\n",
            "Epoch 10/75\n",
            "41/41 [==============================] - 1s 29ms/step - loss: 0.3912 - accuracy: 0.8176\n",
            "Epoch 11/75\n",
            "41/41 [==============================] - 1s 35ms/step - loss: 0.3763 - accuracy: 0.8296\n",
            "Epoch 12/75\n",
            "41/41 [==============================] - 1s 32ms/step - loss: 0.3724 - accuracy: 0.8311\n",
            "Epoch 13/75\n",
            "41/41 [==============================] - 1s 32ms/step - loss: 0.3616 - accuracy: 0.8346\n",
            "Epoch 14/75\n",
            "41/41 [==============================] - 1s 33ms/step - loss: 0.3480 - accuracy: 0.8393\n",
            "Epoch 15/75\n",
            "41/41 [==============================] - 1s 31ms/step - loss: 0.3433 - accuracy: 0.8466\n",
            "Epoch 16/75\n",
            "41/41 [==============================] - 2s 37ms/step - loss: 0.3310 - accuracy: 0.8532\n",
            "Epoch 17/75\n",
            "41/41 [==============================] - 2s 58ms/step - loss: 0.3204 - accuracy: 0.8590\n",
            "Epoch 18/75\n",
            "41/41 [==============================] - 2s 37ms/step - loss: 0.3134 - accuracy: 0.8575\n",
            "Epoch 19/75\n",
            "41/41 [==============================] - 1s 33ms/step - loss: 0.3172 - accuracy: 0.8613\n",
            "Epoch 20/75\n",
            "41/41 [==============================] - 1s 35ms/step - loss: 0.2992 - accuracy: 0.8641\n",
            "Epoch 21/75\n",
            "41/41 [==============================] - 2s 41ms/step - loss: 0.3013 - accuracy: 0.8672\n",
            "Epoch 22/75\n",
            "41/41 [==============================] - 2s 44ms/step - loss: 0.2757 - accuracy: 0.8726\n",
            "Epoch 23/75\n",
            "41/41 [==============================] - 2s 40ms/step - loss: 0.2674 - accuracy: 0.8811\n",
            "Epoch 24/75\n",
            "41/41 [==============================] - 2s 61ms/step - loss: 0.2606 - accuracy: 0.8834\n",
            "Epoch 25/75\n",
            "41/41 [==============================] - 2s 48ms/step - loss: 0.2488 - accuracy: 0.8912\n",
            "Epoch 26/75\n",
            "41/41 [==============================] - 1s 35ms/step - loss: 0.2446 - accuracy: 0.9005\n",
            "Epoch 27/75\n",
            "41/41 [==============================] - 1s 36ms/step - loss: 0.2303 - accuracy: 0.9055\n",
            "Epoch 28/75\n",
            "41/41 [==============================] - 1s 34ms/step - loss: 0.2186 - accuracy: 0.9070\n",
            "Epoch 29/75\n",
            "41/41 [==============================] - 1s 36ms/step - loss: 0.2163 - accuracy: 0.9086\n",
            "Epoch 30/75\n",
            "41/41 [==============================] - 1s 35ms/step - loss: 0.1962 - accuracy: 0.9260\n",
            "Epoch 31/75\n",
            "41/41 [==============================] - 2s 37ms/step - loss: 0.1955 - accuracy: 0.9245\n",
            "Epoch 32/75\n",
            "41/41 [==============================] - 2s 43ms/step - loss: 0.1822 - accuracy: 0.9345\n",
            "Epoch 33/75\n",
            "41/41 [==============================] - 2s 51ms/step - loss: 0.1712 - accuracy: 0.9411\n",
            "Epoch 34/75\n",
            "41/41 [==============================] - 1s 36ms/step - loss: 0.1657 - accuracy: 0.9404\n",
            "Epoch 35/75\n",
            "41/41 [==============================] - 1s 33ms/step - loss: 0.1621 - accuracy: 0.9419\n",
            "Epoch 36/75\n",
            "41/41 [==============================] - 2s 44ms/step - loss: 0.1496 - accuracy: 0.9489\n",
            "Epoch 37/75\n",
            "41/41 [==============================] - 2s 42ms/step - loss: 0.1357 - accuracy: 0.9570\n",
            "Epoch 38/75\n",
            "41/41 [==============================] - 1s 29ms/step - loss: 0.1297 - accuracy: 0.9617\n",
            "Epoch 39/75\n",
            "41/41 [==============================] - 1s 32ms/step - loss: 0.1209 - accuracy: 0.9675\n",
            "Epoch 40/75\n",
            "41/41 [==============================] - 1s 33ms/step - loss: 0.1176 - accuracy: 0.9713\n",
            "Epoch 41/75\n",
            "41/41 [==============================] - 2s 43ms/step - loss: 0.1164 - accuracy: 0.9706\n",
            "Epoch 42/75\n",
            "41/41 [==============================] - 2s 40ms/step - loss: 0.1289 - accuracy: 0.9605\n",
            "Epoch 43/75\n",
            "41/41 [==============================] - 1s 33ms/step - loss: 0.1106 - accuracy: 0.9717\n",
            "Epoch 44/75\n",
            "41/41 [==============================] - 1s 33ms/step - loss: 0.1035 - accuracy: 0.9768\n",
            "Epoch 45/75\n",
            "41/41 [==============================] - 1s 32ms/step - loss: 0.1069 - accuracy: 0.9771\n",
            "Epoch 46/75\n",
            "41/41 [==============================] - 1s 32ms/step - loss: 0.1054 - accuracy: 0.9733\n",
            "Epoch 47/75\n",
            "41/41 [==============================] - 1s 33ms/step - loss: 0.1002 - accuracy: 0.9756\n",
            "Epoch 48/75\n",
            "41/41 [==============================] - 1s 32ms/step - loss: 0.0762 - accuracy: 0.9880\n",
            "Epoch 49/75\n",
            "41/41 [==============================] - 1s 20ms/step - loss: 0.0742 - accuracy: 0.9888\n",
            "Epoch 50/75\n",
            "41/41 [==============================] - 1s 19ms/step - loss: 0.0737 - accuracy: 0.9895\n",
            "Epoch 51/75\n",
            "41/41 [==============================] - 1s 29ms/step - loss: 0.0670 - accuracy: 0.9907\n",
            "Epoch 52/75\n",
            "41/41 [==============================] - 1s 31ms/step - loss: 0.0620 - accuracy: 0.9926\n",
            "Epoch 53/75\n",
            "41/41 [==============================] - 1s 24ms/step - loss: 0.0606 - accuracy: 0.9934\n",
            "Epoch 54/75\n",
            "41/41 [==============================] - 1s 19ms/step - loss: 0.0613 - accuracy: 0.9934\n",
            "Epoch 55/75\n",
            "41/41 [==============================] - 1s 19ms/step - loss: 0.0576 - accuracy: 0.9942\n",
            "Epoch 56/75\n",
            "41/41 [==============================] - 1s 19ms/step - loss: 0.0734 - accuracy: 0.9888\n",
            "Epoch 57/75\n",
            "41/41 [==============================] - 1s 19ms/step - loss: 0.0641 - accuracy: 0.9895\n",
            "Epoch 58/75\n",
            "41/41 [==============================] - 1s 19ms/step - loss: 0.0775 - accuracy: 0.9880\n",
            "Epoch 59/75\n",
            "41/41 [==============================] - 1s 19ms/step - loss: 0.0672 - accuracy: 0.9892\n",
            "Epoch 60/75\n",
            "41/41 [==============================] - 1s 19ms/step - loss: 0.0590 - accuracy: 0.9915\n",
            "Epoch 61/75\n",
            "41/41 [==============================] - 1s 18ms/step - loss: 0.0632 - accuracy: 0.9899\n",
            "Epoch 62/75\n",
            "41/41 [==============================] - 1s 19ms/step - loss: 0.0879 - accuracy: 0.9806\n",
            "Epoch 63/75\n",
            "41/41 [==============================] - 1s 18ms/step - loss: 0.0769 - accuracy: 0.9818\n",
            "Epoch 64/75\n",
            "41/41 [==============================] - 1s 19ms/step - loss: 0.0612 - accuracy: 0.9880\n",
            "Epoch 65/75\n",
            "41/41 [==============================] - 1s 24ms/step - loss: 0.0530 - accuracy: 0.9923\n",
            "Epoch 66/75\n",
            "41/41 [==============================] - 2s 44ms/step - loss: 0.0481 - accuracy: 0.9934\n",
            "Epoch 67/75\n",
            "41/41 [==============================] - 2s 43ms/step - loss: 0.0468 - accuracy: 0.9954\n",
            "Epoch 68/75\n",
            "41/41 [==============================] - 2s 41ms/step - loss: 0.0425 - accuracy: 0.9950\n",
            "Epoch 69/75\n",
            "41/41 [==============================] - 2s 40ms/step - loss: 0.0448 - accuracy: 0.9954\n",
            "Epoch 70/75\n",
            "41/41 [==============================] - 1s 33ms/step - loss: 0.0394 - accuracy: 0.9961\n",
            "Epoch 71/75\n",
            "41/41 [==============================] - 1s 35ms/step - loss: 0.0430 - accuracy: 0.9946\n",
            "Epoch 72/75\n",
            "41/41 [==============================] - 1s 34ms/step - loss: 0.0405 - accuracy: 0.9957\n",
            "Epoch 73/75\n",
            "41/41 [==============================] - 1s 34ms/step - loss: 0.0384 - accuracy: 0.9961\n",
            "Epoch 74/75\n",
            "41/41 [==============================] - 2s 40ms/step - loss: 0.0368 - accuracy: 0.9969\n",
            "Epoch 75/75\n",
            "41/41 [==============================] - 2s 52ms/step - loss: 0.0397 - accuracy: 0.9957\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x794cb749a680>"
            ]
          },
          "metadata": {},
          "execution_count": 158
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = ann.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "o4QBhkEMRHoU",
        "outputId": "73eca84e-5ed9-4e3b-d45f-2702eff0d62b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "21/21 [==============================] - 1s 8ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8uSOKJdARLL0",
        "outputId": "bd48dd56-7af1-4062-d44b-9566385a53d3"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9488372093023256,\n",
              " 0.9552238805970149,\n",
              " 0.8258064516129032,\n",
              " 0.8530700997480413,\n",
              " 0.856823672695163)"
            ]
          },
          "metadata": {},
          "execution_count": 160
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "VEoCa-f3RPdf",
        "outputId": "eec4edda-18d6-4587-f99f-97af368ce28b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9471624266144814"
            ]
          },
          "metadata": {},
          "execution_count": 161
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Test**"
      ],
      "metadata": {
        "id": "fUyWmOUCDyqH"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/NMB-TR.csv')\n",
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "RMyosYFMD0oI"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.3, random_state = 1)"
      ],
      "metadata": {
        "id": "RPKj1OHgn0KE"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "sample_size = xtrain.shape[0] # number of samples in train set\n",
        "time_steps  = xtrain.shape[1] # number of features in train set\n",
        "input_dimension = 1               # each feature is represented by 1 number"
      ],
      "metadata": {
        "id": "qQjCM9e4D72o"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_data_reshaped = xtrain.values.reshape(sample_size,time_steps,input_dimension)\n",
        "n_timesteps = train_data_reshaped.shape[1]\n",
        "n_features  = train_data_reshaped.shape[2]"
      ],
      "metadata": {
        "id": "hSF4-4paD-RS"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann = Sequential()"
      ],
      "metadata": {
        "id": "b-HyjJDgEAMs"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Dense(256, activation = 'relu', input_shape=(n_timesteps,n_features)))\n",
        "ann.add(Dense(256, activation = 'relu'))\n",
        "ann.add(Dense(128, activation = 'relu'))"
      ],
      "metadata": {
        "id": "t6e_UYJzEB5i"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "RahVgj3TED6c"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Flatten())"
      ],
      "metadata": {
        "id": "qIijuW8NEGYp"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Dense(64, activation='relu'))\n",
        "ann.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "_MlUlYkNEIWo"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "U_Vyq8_fEKYJ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.fit(xtrain, ytrain, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bMm8MN-TEMgt",
        "outputId": "a9a1e261-58da-41c3-d457-6c6fee4f9423"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "36/36 [==============================] - 4s 39ms/step - loss: 0.5260 - accuracy: 0.7976\n",
            "Epoch 2/75\n",
            "36/36 [==============================] - 2s 43ms/step - loss: 0.4813 - accuracy: 0.8127\n",
            "Epoch 3/75\n",
            "36/36 [==============================] - 2s 58ms/step - loss: 0.4722 - accuracy: 0.8136\n",
            "Epoch 4/75\n",
            "36/36 [==============================] - 2s 47ms/step - loss: 0.4454 - accuracy: 0.8149\n",
            "Epoch 5/75\n",
            "36/36 [==============================] - 1s 39ms/step - loss: 0.4329 - accuracy: 0.8158\n",
            "Epoch 6/75\n",
            "36/36 [==============================] - 1s 39ms/step - loss: 0.4252 - accuracy: 0.8198\n",
            "Epoch 7/75\n",
            "36/36 [==============================] - 1s 38ms/step - loss: 0.4160 - accuracy: 0.8220\n",
            "Epoch 8/75\n",
            "36/36 [==============================] - 1s 38ms/step - loss: 0.4135 - accuracy: 0.8264\n",
            "Epoch 9/75\n",
            "36/36 [==============================] - 1s 39ms/step - loss: 0.4068 - accuracy: 0.8202\n",
            "Epoch 10/75\n",
            "36/36 [==============================] - 1s 38ms/step - loss: 0.4006 - accuracy: 0.8246\n",
            "Epoch 11/75\n",
            "36/36 [==============================] - 2s 48ms/step - loss: 0.3945 - accuracy: 0.8295\n",
            "Epoch 12/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.3878 - accuracy: 0.8397\n",
            "Epoch 13/75\n",
            "36/36 [==============================] - 2s 45ms/step - loss: 0.3823 - accuracy: 0.8463\n",
            "Epoch 14/75\n",
            "36/36 [==============================] - 2s 45ms/step - loss: 0.3723 - accuracy: 0.8454\n",
            "Epoch 15/75\n",
            "36/36 [==============================] - 1s 39ms/step - loss: 0.3691 - accuracy: 0.8508\n",
            "Epoch 16/75\n",
            "36/36 [==============================] - 1s 40ms/step - loss: 0.3638 - accuracy: 0.8490\n",
            "Epoch 17/75\n",
            "36/36 [==============================] - 1s 39ms/step - loss: 0.3672 - accuracy: 0.8583\n",
            "Epoch 18/75\n",
            "36/36 [==============================] - 1s 38ms/step - loss: 0.3508 - accuracy: 0.8618\n",
            "Epoch 19/75\n",
            "36/36 [==============================] - 1s 40ms/step - loss: 0.3449 - accuracy: 0.8640\n",
            "Epoch 20/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.3338 - accuracy: 0.8716\n",
            "Epoch 21/75\n",
            "36/36 [==============================] - 2s 52ms/step - loss: 0.3245 - accuracy: 0.8756\n",
            "Epoch 22/75\n",
            "36/36 [==============================] - 1s 39ms/step - loss: 0.3208 - accuracy: 0.8778\n",
            "Epoch 23/75\n",
            "36/36 [==============================] - 2s 47ms/step - loss: 0.3136 - accuracy: 0.8720\n",
            "Epoch 24/75\n",
            "36/36 [==============================] - 1s 38ms/step - loss: 0.2908 - accuracy: 0.8826\n",
            "Epoch 25/75\n",
            "36/36 [==============================] - 1s 41ms/step - loss: 0.2805 - accuracy: 0.8959\n",
            "Epoch 26/75\n",
            "36/36 [==============================] - 1s 39ms/step - loss: 0.2707 - accuracy: 0.9008\n",
            "Epoch 27/75\n",
            "36/36 [==============================] - 1s 40ms/step - loss: 0.2625 - accuracy: 0.8959\n",
            "Epoch 28/75\n",
            "36/36 [==============================] - 2s 50ms/step - loss: 0.2509 - accuracy: 0.9030\n",
            "Epoch 29/75\n",
            "36/36 [==============================] - 2s 60ms/step - loss: 0.2485 - accuracy: 0.9092\n",
            "Epoch 30/75\n",
            "36/36 [==============================] - 2s 43ms/step - loss: 0.2364 - accuracy: 0.9123\n",
            "Epoch 31/75\n",
            "36/36 [==============================] - 1s 38ms/step - loss: 0.2241 - accuracy: 0.9097\n",
            "Epoch 32/75\n",
            "36/36 [==============================] - 1s 37ms/step - loss: 0.2181 - accuracy: 0.9185\n",
            "Epoch 33/75\n",
            "36/36 [==============================] - 1s 37ms/step - loss: 0.2026 - accuracy: 0.9212\n",
            "Epoch 34/75\n",
            "36/36 [==============================] - 1s 37ms/step - loss: 0.1968 - accuracy: 0.9229\n",
            "Epoch 35/75\n",
            "36/36 [==============================] - 1s 37ms/step - loss: 0.2015 - accuracy: 0.9198\n",
            "Epoch 36/75\n",
            "36/36 [==============================] - 1s 39ms/step - loss: 0.1955 - accuracy: 0.9265\n",
            "Epoch 37/75\n",
            "36/36 [==============================] - 2s 48ms/step - loss: 0.1711 - accuracy: 0.9402\n",
            "Epoch 38/75\n",
            "36/36 [==============================] - 2s 63ms/step - loss: 0.1719 - accuracy: 0.9376\n",
            "Epoch 39/75\n",
            "36/36 [==============================] - 2s 43ms/step - loss: 0.1658 - accuracy: 0.9358\n",
            "Epoch 40/75\n",
            "36/36 [==============================] - 1s 36ms/step - loss: 0.1526 - accuracy: 0.9464\n",
            "Epoch 41/75\n",
            "36/36 [==============================] - 1s 36ms/step - loss: 0.1364 - accuracy: 0.9522\n",
            "Epoch 42/75\n",
            "36/36 [==============================] - 1s 35ms/step - loss: 0.1266 - accuracy: 0.9513\n",
            "Epoch 43/75\n",
            "36/36 [==============================] - 1s 36ms/step - loss: 0.1275 - accuracy: 0.9500\n",
            "Epoch 44/75\n",
            "36/36 [==============================] - 1s 31ms/step - loss: 0.1330 - accuracy: 0.9473\n",
            "Epoch 45/75\n",
            "36/36 [==============================] - 1s 37ms/step - loss: 0.1241 - accuracy: 0.9548\n",
            "Epoch 46/75\n",
            "36/36 [==============================] - 1s 39ms/step - loss: 0.1058 - accuracy: 0.9610\n",
            "Epoch 47/75\n",
            "36/36 [==============================] - 2s 56ms/step - loss: 0.1062 - accuracy: 0.9606\n",
            "Epoch 48/75\n",
            "36/36 [==============================] - 2s 42ms/step - loss: 0.0895 - accuracy: 0.9690\n",
            "Epoch 49/75\n",
            "36/36 [==============================] - 1s 35ms/step - loss: 0.0891 - accuracy: 0.9681\n",
            "Epoch 50/75\n",
            "36/36 [==============================] - 1s 36ms/step - loss: 0.0968 - accuracy: 0.9672\n",
            "Epoch 51/75\n",
            "36/36 [==============================] - 1s 37ms/step - loss: 0.0754 - accuracy: 0.9783\n",
            "Epoch 52/75\n",
            "36/36 [==============================] - 1s 35ms/step - loss: 0.0983 - accuracy: 0.9628\n",
            "Epoch 53/75\n",
            "36/36 [==============================] - 1s 35ms/step - loss: 0.0823 - accuracy: 0.9739\n",
            "Epoch 54/75\n",
            "36/36 [==============================] - 1s 35ms/step - loss: 0.0659 - accuracy: 0.9814\n",
            "Epoch 55/75\n",
            "36/36 [==============================] - 1s 38ms/step - loss: 0.0643 - accuracy: 0.9787\n",
            "Epoch 56/75\n",
            "36/36 [==============================] - 2s 57ms/step - loss: 0.0597 - accuracy: 0.9841\n",
            "Epoch 57/75\n",
            "36/36 [==============================] - 2s 46ms/step - loss: 0.0670 - accuracy: 0.9796\n",
            "Epoch 58/75\n",
            "36/36 [==============================] - 2s 44ms/step - loss: 0.0543 - accuracy: 0.9849\n",
            "Epoch 59/75\n",
            "36/36 [==============================] - 1s 34ms/step - loss: 0.0541 - accuracy: 0.9845\n",
            "Epoch 60/75\n",
            "36/36 [==============================] - 1s 36ms/step - loss: 0.0393 - accuracy: 0.9903\n",
            "Epoch 61/75\n",
            "36/36 [==============================] - 1s 35ms/step - loss: 0.0338 - accuracy: 0.9938\n",
            "Epoch 62/75\n",
            "36/36 [==============================] - 1s 33ms/step - loss: 0.0317 - accuracy: 0.9925\n",
            "Epoch 63/75\n",
            "36/36 [==============================] - 1s 37ms/step - loss: 0.0289 - accuracy: 0.9942\n",
            "Epoch 64/75\n",
            "36/36 [==============================] - 1s 36ms/step - loss: 0.0477 - accuracy: 0.9832\n",
            "Epoch 65/75\n",
            "36/36 [==============================] - 1s 41ms/step - loss: 0.0545 - accuracy: 0.9832\n",
            "Epoch 66/75\n",
            "36/36 [==============================] - 2s 48ms/step - loss: 0.0631 - accuracy: 0.9774\n",
            "Epoch 67/75\n",
            "36/36 [==============================] - 2s 43ms/step - loss: 0.0763 - accuracy: 0.9748\n",
            "Epoch 68/75\n",
            "36/36 [==============================] - 1s 36ms/step - loss: 0.0553 - accuracy: 0.9823\n",
            "Epoch 69/75\n",
            "36/36 [==============================] - 1s 38ms/step - loss: 0.0411 - accuracy: 0.9863\n",
            "Epoch 70/75\n",
            "36/36 [==============================] - 1s 37ms/step - loss: 0.0242 - accuracy: 0.9956\n",
            "Epoch 71/75\n",
            "36/36 [==============================] - 1s 36ms/step - loss: 0.0266 - accuracy: 0.9925\n",
            "Epoch 72/75\n",
            "36/36 [==============================] - 1s 37ms/step - loss: 0.0305 - accuracy: 0.9920\n",
            "Epoch 73/75\n",
            "36/36 [==============================] - 1s 35ms/step - loss: 0.0207 - accuracy: 0.9947\n",
            "Epoch 74/75\n",
            "36/36 [==============================] - 1s 37ms/step - loss: 0.0153 - accuracy: 0.9969\n",
            "Epoch 75/75\n",
            "36/36 [==============================] - 1s 41ms/step - loss: 0.0173 - accuracy: 0.9951\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7e1f74359720>"
            ]
          },
          "metadata": {},
          "execution_count": 106
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = ann.predict(xtest)\n",
        "pred = (pred > 0.5)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_49eqdQfEOXq",
        "outputId": "ab4913c5-53c2-4e8c-ca77-e9c971239653"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "31/31 [==============================] - 0s 5ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(ytest, pred), precision_score(ytest, pred), recall_score(ytest, pred), f1_score(ytest, pred), cohen_kappa_score(ytest, pred), matthews_corrcoef(ytest, pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "kTlTm3dtEQ3j",
        "outputId": "c1496132-be6f-43ce-fed7-013deaf5cf46"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9215686274509803,\n",
              " 0.7943548387096774,\n",
              " 0.8873873873873874,\n",
              " 0.8382978723404255,\n",
              " 0.786735704481087,\n",
              " 0.7888379950145482)"
            ]
          },
          "metadata": {},
          "execution_count": 108
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(ytest, pred)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "BYlVGs7nETEx",
        "outputId": "26b6b6e6-9245-4a34-92c6-4f881016baa3"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9317269076305221"
            ]
          },
          "metadata": {},
          "execution_count": 109
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**ADASYN**"
      ],
      "metadata": {
        "id": "IcD8sAaLRTOr"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/NMB-TR.csv')"
      ],
      "metadata": {
        "id": "tf9ypeGfRWC7"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "FtAJRrgeRfTb"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.over_sampling import ADASYN\n",
        "ada = ADASYN()\n",
        "X, Y = ada.fit_resample(X, Y)"
      ],
      "metadata": {
        "id": "n-icLOWHRhpH"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ],
      "metadata": {
        "id": "7FmJIbfeRk37"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ],
      "metadata": {
        "id": "Uk-xo79uRnBb"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann = Sequential()"
      ],
      "metadata": {
        "id": "36uKrO18RpNj"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Dense(256, activation = 'relu', input_shape=(X_train.shape[1], 1)))\n",
        "ann.add(Dense(256, activation = 'relu'))\n",
        "ann.add(Dense(128, activation = 'relu'))"
      ],
      "metadata": {
        "id": "uvCD3Z_vRrbw"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "Z7d3OzvcRtq0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Flatten())"
      ],
      "metadata": {
        "id": "WjGDrLhYRwCF"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Dense(64, activation='relu'))\n",
        "ann.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "VJNqcQNORx9V"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "hSSKHCDOR0LP"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "PgUglvgRR2Ps",
        "outputId": "63c40c1c-46b2-4b77-9d4a-5be0cc2bc940"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "66/66 [==============================] - 3s 19ms/step - loss: 0.6756 - accuracy: 0.5750\n",
            "Epoch 2/75\n",
            "66/66 [==============================] - 1s 19ms/step - loss: 0.6353 - accuracy: 0.6458\n",
            "Epoch 3/75\n",
            "66/66 [==============================] - 1s 19ms/step - loss: 0.5694 - accuracy: 0.7142\n",
            "Epoch 4/75\n",
            "66/66 [==============================] - 1s 19ms/step - loss: 0.4954 - accuracy: 0.7700\n",
            "Epoch 5/75\n",
            "66/66 [==============================] - 1s 19ms/step - loss: 0.4180 - accuracy: 0.8083\n",
            "Epoch 6/75\n",
            "66/66 [==============================] - 2s 23ms/step - loss: 0.3478 - accuracy: 0.8584\n",
            "Epoch 7/75\n",
            "66/66 [==============================] - 2s 32ms/step - loss: 0.3168 - accuracy: 0.8746\n",
            "Epoch 8/75\n",
            "66/66 [==============================] - 1s 19ms/step - loss: 0.2651 - accuracy: 0.8992\n",
            "Epoch 9/75\n",
            "66/66 [==============================] - 1s 19ms/step - loss: 0.2347 - accuracy: 0.9147\n",
            "Epoch 10/75\n",
            "66/66 [==============================] - 1s 19ms/step - loss: 0.1882 - accuracy: 0.9309\n",
            "Epoch 11/75\n",
            "66/66 [==============================] - 1s 19ms/step - loss: 0.1610 - accuracy: 0.9442\n",
            "Epoch 12/75\n",
            "66/66 [==============================] - 1s 19ms/step - loss: 0.1460 - accuracy: 0.9502\n",
            "Epoch 13/75\n",
            "66/66 [==============================] - 1s 19ms/step - loss: 0.1430 - accuracy: 0.9502\n",
            "Epoch 14/75\n",
            "66/66 [==============================] - 1s 18ms/step - loss: 0.1123 - accuracy: 0.9631\n",
            "Epoch 15/75\n",
            "66/66 [==============================] - 1s 19ms/step - loss: 0.1004 - accuracy: 0.9659\n",
            "Epoch 16/75\n",
            "66/66 [==============================] - 2s 31ms/step - loss: 0.1027 - accuracy: 0.9664\n",
            "Epoch 17/75\n",
            "66/66 [==============================] - 2s 24ms/step - loss: 0.0878 - accuracy: 0.9709\n",
            "Epoch 18/75\n",
            "66/66 [==============================] - 1s 19ms/step - loss: 0.0674 - accuracy: 0.9788\n",
            "Epoch 19/75\n",
            "66/66 [==============================] - 2s 33ms/step - loss: 0.0604 - accuracy: 0.9824\n",
            "Epoch 20/75\n",
            "66/66 [==============================] - 2s 33ms/step - loss: 0.0876 - accuracy: 0.9695\n",
            "Epoch 21/75\n",
            "66/66 [==============================] - 2s 32ms/step - loss: 0.0503 - accuracy: 0.9840\n",
            "Epoch 22/75\n",
            "66/66 [==============================] - 3s 39ms/step - loss: 0.0389 - accuracy: 0.9890\n",
            "Epoch 23/75\n",
            "66/66 [==============================] - 3s 39ms/step - loss: 0.0372 - accuracy: 0.9890\n",
            "Epoch 24/75\n",
            "66/66 [==============================] - 2s 28ms/step - loss: 0.0585 - accuracy: 0.9800\n",
            "Epoch 25/75\n",
            "66/66 [==============================] - 2s 27ms/step - loss: 0.0643 - accuracy: 0.9800\n",
            "Epoch 26/75\n",
            "66/66 [==============================] - 2s 24ms/step - loss: 0.0260 - accuracy: 0.9940\n",
            "Epoch 27/75\n",
            "66/66 [==============================] - 2s 32ms/step - loss: 0.0248 - accuracy: 0.9948\n",
            "Epoch 28/75\n",
            "66/66 [==============================] - 2s 32ms/step - loss: 0.0258 - accuracy: 0.9931\n",
            "Epoch 29/75\n",
            "66/66 [==============================] - 3s 43ms/step - loss: 0.0322 - accuracy: 0.9909\n",
            "Epoch 30/75\n",
            "66/66 [==============================] - 2s 34ms/step - loss: 0.0505 - accuracy: 0.9845\n",
            "Epoch 31/75\n",
            "66/66 [==============================] - 3s 40ms/step - loss: 0.0520 - accuracy: 0.9855\n",
            "Epoch 32/75\n",
            "66/66 [==============================] - 2s 35ms/step - loss: 0.0307 - accuracy: 0.9914\n",
            "Epoch 33/75\n",
            "66/66 [==============================] - 2s 35ms/step - loss: 0.0127 - accuracy: 0.9979\n",
            "Epoch 34/75\n",
            "66/66 [==============================] - 3s 44ms/step - loss: 0.0425 - accuracy: 0.9888\n",
            "Epoch 35/75\n",
            "66/66 [==============================] - 3s 42ms/step - loss: 0.0301 - accuracy: 0.9924\n",
            "Epoch 36/75\n",
            "66/66 [==============================] - 2s 34ms/step - loss: 0.0157 - accuracy: 0.9967\n",
            "Epoch 37/75\n",
            "66/66 [==============================] - 2s 36ms/step - loss: 0.0081 - accuracy: 0.9988\n",
            "Epoch 38/75\n",
            "66/66 [==============================] - 2s 36ms/step - loss: 0.0116 - accuracy: 0.9976\n",
            "Epoch 39/75\n",
            "66/66 [==============================] - 3s 40ms/step - loss: 0.0151 - accuracy: 0.9959\n",
            "Epoch 40/75\n",
            "66/66 [==============================] - 3s 48ms/step - loss: 0.0265 - accuracy: 0.9924\n",
            "Epoch 41/75\n",
            "66/66 [==============================] - 2s 34ms/step - loss: 0.0107 - accuracy: 0.9981\n",
            "Epoch 42/75\n",
            "66/66 [==============================] - 2s 33ms/step - loss: 0.0125 - accuracy: 0.9969\n",
            "Epoch 43/75\n",
            "66/66 [==============================] - 2s 30ms/step - loss: 0.0215 - accuracy: 0.9940\n",
            "Epoch 44/75\n",
            "66/66 [==============================] - 2s 32ms/step - loss: 0.0431 - accuracy: 0.9876\n",
            "Epoch 45/75\n",
            "66/66 [==============================] - 2s 36ms/step - loss: 0.0677 - accuracy: 0.9795\n",
            "Epoch 46/75\n",
            "66/66 [==============================] - 2s 36ms/step - loss: 0.0311 - accuracy: 0.9900\n",
            "Epoch 47/75\n",
            "66/66 [==============================] - 2s 31ms/step - loss: 0.0124 - accuracy: 0.9979\n",
            "Epoch 48/75\n",
            "66/66 [==============================] - 2s 33ms/step - loss: 0.0107 - accuracy: 0.9974\n",
            "Epoch 49/75\n",
            "66/66 [==============================] - 2s 33ms/step - loss: 0.0115 - accuracy: 0.9974\n",
            "Epoch 50/75\n",
            "66/66 [==============================] - 2s 36ms/step - loss: 0.0108 - accuracy: 0.9979\n",
            "Epoch 51/75\n",
            "66/66 [==============================] - 3s 43ms/step - loss: 0.0092 - accuracy: 0.9976\n",
            "Epoch 52/75\n",
            "66/66 [==============================] - 2s 35ms/step - loss: 0.0080 - accuracy: 0.9981\n",
            "Epoch 53/75\n",
            "66/66 [==============================] - 2s 36ms/step - loss: 0.0078 - accuracy: 0.9983\n",
            "Epoch 54/75\n",
            "66/66 [==============================] - 3s 43ms/step - loss: 0.0070 - accuracy: 0.9988\n",
            "Epoch 55/75\n",
            "66/66 [==============================] - 2s 34ms/step - loss: 0.0231 - accuracy: 0.9948\n",
            "Epoch 56/75\n",
            "66/66 [==============================] - 3s 48ms/step - loss: 0.0462 - accuracy: 0.9864\n",
            "Epoch 57/75\n",
            "66/66 [==============================] - 3s 39ms/step - loss: 0.0268 - accuracy: 0.9924\n",
            "Epoch 58/75\n",
            "66/66 [==============================] - 2s 36ms/step - loss: 0.0413 - accuracy: 0.9843\n",
            "Epoch 59/75\n",
            "66/66 [==============================] - 2s 35ms/step - loss: 0.0468 - accuracy: 0.9878\n",
            "Epoch 60/75\n",
            "66/66 [==============================] - 2s 35ms/step - loss: 0.0827 - accuracy: 0.9745\n",
            "Epoch 61/75\n",
            "66/66 [==============================] - 3s 42ms/step - loss: 0.0381 - accuracy: 0.9890\n",
            "Epoch 62/75\n",
            "66/66 [==============================] - 3s 46ms/step - loss: 0.0150 - accuracy: 0.9971\n",
            "Epoch 63/75\n",
            "66/66 [==============================] - 2s 28ms/step - loss: 0.0074 - accuracy: 0.9988\n",
            "Epoch 64/75\n",
            "66/66 [==============================] - 2s 32ms/step - loss: 0.0084 - accuracy: 0.9981\n",
            "Epoch 65/75\n",
            "66/66 [==============================] - 2s 31ms/step - loss: 0.0068 - accuracy: 0.9990\n",
            "Epoch 66/75\n",
            "66/66 [==============================] - 2s 31ms/step - loss: 0.0114 - accuracy: 0.9974\n",
            "Epoch 67/75\n",
            "66/66 [==============================] - 2s 36ms/step - loss: 0.0086 - accuracy: 0.9983\n",
            "Epoch 68/75\n",
            "66/66 [==============================] - 2s 36ms/step - loss: 0.0140 - accuracy: 0.9967\n",
            "Epoch 69/75\n",
            "66/66 [==============================] - 2s 33ms/step - loss: 0.0103 - accuracy: 0.9976\n",
            "Epoch 70/75\n",
            "66/66 [==============================] - 2s 30ms/step - loss: 0.0106 - accuracy: 0.9979\n",
            "Epoch 71/75\n",
            "66/66 [==============================] - 2s 29ms/step - loss: 0.0072 - accuracy: 0.9993\n",
            "Epoch 72/75\n",
            "66/66 [==============================] - 2s 36ms/step - loss: 0.0147 - accuracy: 0.9969\n",
            "Epoch 73/75\n",
            "66/66 [==============================] - 3s 42ms/step - loss: 0.0123 - accuracy: 0.9986\n",
            "Epoch 74/75\n",
            "66/66 [==============================] - 2s 36ms/step - loss: 0.0065 - accuracy: 0.9986\n",
            "Epoch 75/75\n",
            "66/66 [==============================] - 2s 30ms/step - loss: 0.0083 - accuracy: 0.9986\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x794cb37bdff0>"
            ]
          },
          "metadata": {},
          "execution_count": 187
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = ann.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "I83xWfEbR4Pa",
        "outputId": "84d94da1-174f-4b3c-e055-8b40fd0b88d3"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "33/33 [==============================] - 0s 7ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "c186ghR5R8Cn",
        "outputId": "a099995e-4ebe-4f12-cb2a-550d5bd6242a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9637404580152672,\n",
              " 0.9943181818181818,\n",
              " 0.9375,\n",
              " 0.9274428638059702,\n",
              " 0.929178938062839)"
            ]
          },
          "metadata": {},
          "execution_count": 189
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "dYGJMnr6R_XL",
        "outputId": "2907196c-6418-4abb-8463-0da48704332a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9326923076923077"
            ]
          },
          "metadata": {},
          "execution_count": 190
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**SMOTETomek**"
      ],
      "metadata": {
        "id": "PvRUyjw-SCPD"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/NMB-TR.csv')"
      ],
      "metadata": {
        "id": "u-B4ia83SE3t"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "pMgEW4TgSKmL"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.combine import SMOTETomek\n",
        "smt = SMOTETomek()\n",
        "X, Y = smt.fit_resample(X, Y)"
      ],
      "metadata": {
        "id": "kjWDliKmSNAo"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ],
      "metadata": {
        "id": "pG0Nxzl9SP6V"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ],
      "metadata": {
        "id": "09xpdqzwSSUD"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann = Sequential()"
      ],
      "metadata": {
        "id": "O0SJfJxUSUvr"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Dense(256, activation = 'relu', input_shape=(X_train.shape[1], 1)))\n",
        "ann.add(Dense(256, activation = 'relu'))\n",
        "ann.add(Dense(128, activation = 'relu'))"
      ],
      "metadata": {
        "id": "GcYFf0rUSW_K"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "dnK5lDgESZua"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Flatten())"
      ],
      "metadata": {
        "id": "zJrx_58BScGt"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Dense(64, activation='relu'))\n",
        "ann.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "ixuQAHKNSgvr"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "_k8lJq62Sh1H"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "dE5oeP5iSkd6",
        "outputId": "24c5fa43-fb6c-4820-f216-42321bb61f94"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "65/65 [==============================] - 52s 19ms/step - loss: 0.6549 - accuracy: 0.6240\n",
            "Epoch 2/75\n",
            "65/65 [==============================] - 1s 19ms/step - loss: 0.5915 - accuracy: 0.7077\n",
            "Epoch 3/75\n",
            "65/65 [==============================] - 1s 19ms/step - loss: 0.5357 - accuracy: 0.7492\n",
            "Epoch 4/75\n",
            "65/65 [==============================] - 1s 19ms/step - loss: 0.4771 - accuracy: 0.7809\n",
            "Epoch 5/75\n",
            "65/65 [==============================] - 1s 19ms/step - loss: 0.4185 - accuracy: 0.8145\n",
            "Epoch 6/75\n",
            "65/65 [==============================] - 1s 18ms/step - loss: 0.3782 - accuracy: 0.8366\n",
            "Epoch 7/75\n",
            "65/65 [==============================] - 1s 19ms/step - loss: 0.3316 - accuracy: 0.8630\n",
            "Epoch 8/75\n",
            "65/65 [==============================] - 2s 27ms/step - loss: 0.2948 - accuracy: 0.8777\n",
            "Epoch 9/75\n",
            "65/65 [==============================] - 2s 29ms/step - loss: 0.2562 - accuracy: 0.8969\n",
            "Epoch 10/75\n",
            "65/65 [==============================] - 1s 19ms/step - loss: 0.2275 - accuracy: 0.9068\n",
            "Epoch 11/75\n",
            "65/65 [==============================] - 1s 19ms/step - loss: 0.2011 - accuracy: 0.9206\n",
            "Epoch 12/75\n",
            "65/65 [==============================] - 1s 18ms/step - loss: 0.1842 - accuracy: 0.9337\n",
            "Epoch 13/75\n",
            "65/65 [==============================] - 1s 19ms/step - loss: 0.1587 - accuracy: 0.9446\n",
            "Epoch 14/75\n",
            "65/65 [==============================] - 1s 19ms/step - loss: 0.1298 - accuracy: 0.9567\n",
            "Epoch 15/75\n",
            "65/65 [==============================] - 1s 20ms/step - loss: 0.1298 - accuracy: 0.9593\n",
            "Epoch 16/75\n",
            "65/65 [==============================] - 1s 19ms/step - loss: 0.1207 - accuracy: 0.9625\n",
            "Epoch 17/75\n",
            "65/65 [==============================] - 1s 20ms/step - loss: 0.1130 - accuracy: 0.9642\n",
            "Epoch 18/75\n",
            "65/65 [==============================] - 2s 31ms/step - loss: 0.0878 - accuracy: 0.9731\n",
            "Epoch 19/75\n",
            "65/65 [==============================] - 2s 23ms/step - loss: 0.0651 - accuracy: 0.9814\n",
            "Epoch 20/75\n",
            "65/65 [==============================] - 1s 19ms/step - loss: 0.0615 - accuracy: 0.9811\n",
            "Epoch 21/75\n",
            "65/65 [==============================] - 1s 19ms/step - loss: 0.0822 - accuracy: 0.9738\n",
            "Epoch 22/75\n",
            "65/65 [==============================] - 1s 19ms/step - loss: 0.1079 - accuracy: 0.9637\n",
            "Epoch 23/75\n",
            "65/65 [==============================] - 1s 19ms/step - loss: 0.0645 - accuracy: 0.9794\n",
            "Epoch 24/75\n",
            "65/65 [==============================] - 1s 19ms/step - loss: 0.0468 - accuracy: 0.9845\n",
            "Epoch 25/75\n",
            "65/65 [==============================] - 1s 19ms/step - loss: 0.0373 - accuracy: 0.9889\n",
            "Epoch 26/75\n",
            "65/65 [==============================] - 1s 19ms/step - loss: 0.0321 - accuracy: 0.9913\n",
            "Epoch 27/75\n",
            "65/65 [==============================] - 2s 28ms/step - loss: 0.0233 - accuracy: 0.9944\n",
            "Epoch 28/75\n",
            "65/65 [==============================] - 2s 28ms/step - loss: 0.0409 - accuracy: 0.9891\n",
            "Epoch 29/75\n",
            "65/65 [==============================] - 1s 19ms/step - loss: 0.0235 - accuracy: 0.9947\n",
            "Epoch 30/75\n",
            "65/65 [==============================] - 1s 19ms/step - loss: 0.0243 - accuracy: 0.9935\n",
            "Epoch 31/75\n",
            "65/65 [==============================] - 1s 19ms/step - loss: 0.0289 - accuracy: 0.9910\n",
            "Epoch 32/75\n",
            "65/65 [==============================] - 1s 19ms/step - loss: 0.0263 - accuracy: 0.9930\n",
            "Epoch 33/75\n",
            "65/65 [==============================] - 1s 19ms/step - loss: 0.0242 - accuracy: 0.9920\n",
            "Epoch 34/75\n",
            "65/65 [==============================] - 1s 19ms/step - loss: 0.0314 - accuracy: 0.9903\n",
            "Epoch 35/75\n",
            "65/65 [==============================] - 1s 20ms/step - loss: 0.0490 - accuracy: 0.9850\n",
            "Epoch 36/75\n",
            "65/65 [==============================] - 1s 22ms/step - loss: 0.0417 - accuracy: 0.9857\n",
            "Epoch 37/75\n",
            "65/65 [==============================] - 2s 32ms/step - loss: 0.0169 - accuracy: 0.9964\n",
            "Epoch 38/75\n",
            "65/65 [==============================] - 1s 21ms/step - loss: 0.0340 - accuracy: 0.9906\n",
            "Epoch 39/75\n",
            "65/65 [==============================] - 1s 19ms/step - loss: 0.0305 - accuracy: 0.9877\n",
            "Epoch 40/75\n",
            "65/65 [==============================] - 1s 18ms/step - loss: 0.0610 - accuracy: 0.9792\n",
            "Epoch 41/75\n",
            "65/65 [==============================] - 1s 19ms/step - loss: 0.0262 - accuracy: 0.9925\n",
            "Epoch 42/75\n",
            "65/65 [==============================] - 1s 19ms/step - loss: 0.0111 - accuracy: 0.9981\n",
            "Epoch 43/75\n",
            "65/65 [==============================] - 1s 19ms/step - loss: 0.0123 - accuracy: 0.9981\n",
            "Epoch 44/75\n",
            "65/65 [==============================] - 1s 19ms/step - loss: 0.0256 - accuracy: 0.9942\n",
            "Epoch 45/75\n",
            "65/65 [==============================] - 1s 19ms/step - loss: 0.0129 - accuracy: 0.9981\n",
            "Epoch 46/75\n",
            "65/65 [==============================] - 2s 28ms/step - loss: 0.0229 - accuracy: 0.9942\n",
            "Epoch 47/75\n",
            "65/65 [==============================] - 2s 28ms/step - loss: 0.0071 - accuracy: 0.9985\n",
            "Epoch 48/75\n",
            "65/65 [==============================] - 2s 26ms/step - loss: 0.0050 - accuracy: 0.9995\n",
            "Epoch 49/75\n",
            "65/65 [==============================] - 2s 29ms/step - loss: 0.0040 - accuracy: 0.9993\n",
            "Epoch 50/75\n",
            "65/65 [==============================] - 2s 32ms/step - loss: 0.0032 - accuracy: 0.9995\n",
            "Epoch 51/75\n",
            "65/65 [==============================] - 2s 28ms/step - loss: 0.0057 - accuracy: 0.9990\n",
            "Epoch 52/75\n",
            "65/65 [==============================] - 2s 32ms/step - loss: 0.0037 - accuracy: 0.9998\n",
            "Epoch 53/75\n",
            "65/65 [==============================] - 3s 45ms/step - loss: 0.0037 - accuracy: 0.9988\n",
            "Epoch 54/75\n",
            "65/65 [==============================] - 1s 21ms/step - loss: 0.0155 - accuracy: 0.9956\n",
            "Epoch 55/75\n",
            "65/65 [==============================] - 1s 19ms/step - loss: 0.0456 - accuracy: 0.9867\n",
            "Epoch 56/75\n",
            "65/65 [==============================] - 1s 19ms/step - loss: 0.0928 - accuracy: 0.9683\n",
            "Epoch 57/75\n",
            "65/65 [==============================] - 1s 19ms/step - loss: 0.0206 - accuracy: 0.9944\n",
            "Epoch 58/75\n",
            "65/65 [==============================] - 1s 19ms/step - loss: 0.0093 - accuracy: 0.9988\n",
            "Epoch 59/75\n",
            "65/65 [==============================] - 1s 19ms/step - loss: 0.0138 - accuracy: 0.9971\n",
            "Epoch 60/75\n",
            "65/65 [==============================] - 1s 19ms/step - loss: 0.0056 - accuracy: 0.9990\n",
            "Epoch 61/75\n",
            "65/65 [==============================] - 1s 19ms/step - loss: 0.0039 - accuracy: 0.9993\n",
            "Epoch 62/75\n",
            "65/65 [==============================] - 2s 32ms/step - loss: 0.0030 - accuracy: 0.9995\n",
            "Epoch 63/75\n",
            "65/65 [==============================] - 2s 24ms/step - loss: 0.0054 - accuracy: 0.9990\n",
            "Epoch 64/75\n",
            "65/65 [==============================] - 1s 19ms/step - loss: 0.0144 - accuracy: 0.9949\n",
            "Epoch 65/75\n",
            "65/65 [==============================] - 2s 23ms/step - loss: 0.0322 - accuracy: 0.9901\n",
            "Epoch 66/75\n",
            "65/65 [==============================] - 2s 33ms/step - loss: 0.0899 - accuracy: 0.9731\n",
            "Epoch 67/75\n",
            "65/65 [==============================] - 2s 31ms/step - loss: 0.0371 - accuracy: 0.9874\n",
            "Epoch 68/75\n",
            "65/65 [==============================] - 2s 32ms/step - loss: 0.0079 - accuracy: 0.9983\n",
            "Epoch 69/75\n",
            "65/65 [==============================] - 3s 40ms/step - loss: 0.0044 - accuracy: 0.9993\n",
            "Epoch 70/75\n",
            "65/65 [==============================] - 2s 35ms/step - loss: 0.0041 - accuracy: 0.9990\n",
            "Epoch 71/75\n",
            "65/65 [==============================] - 2s 35ms/step - loss: 0.0034 - accuracy: 0.9995\n",
            "Epoch 72/75\n",
            "65/65 [==============================] - 2s 32ms/step - loss: 0.0044 - accuracy: 0.9990\n",
            "Epoch 73/75\n",
            "65/65 [==============================] - 2s 33ms/step - loss: 0.0048 - accuracy: 0.9995\n",
            "Epoch 74/75\n",
            "65/65 [==============================] - 2s 33ms/step - loss: 0.0035 - accuracy: 0.9993\n",
            "Epoch 75/75\n",
            "65/65 [==============================] - 3s 47ms/step - loss: 0.0033 - accuracy: 0.9993\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x794cb3466ec0>"
            ]
          },
          "metadata": {},
          "execution_count": 202
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = ann.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "gxOCXRh9SnDS",
        "outputId": "32652f75-7e34-4fcd-e944-1fe5d6561e0a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "33/33 [==============================] - 0s 7ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "iEb09KvdSqSy",
        "outputId": "45e8c2cf-82c0-4884-8bd3-4cda0f7936a3"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9428294573643411,\n",
              " 1.0,\n",
              " 0.8987993138936535,\n",
              " 0.8854282703597772,\n",
              " 0.8912974837857758)"
            ]
          },
          "metadata": {},
          "execution_count": 204
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7QNa1-zjTC2G",
        "outputId": "08067374-211d-47c2-aa22-e67401d0c33b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.8838582677165354"
            ]
          },
          "metadata": {},
          "execution_count": 205
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**NearMiss**"
      ],
      "metadata": {
        "id": "lmEfQK0mTFVs"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/NMB-TR.csv')"
      ],
      "metadata": {
        "id": "6ApJ9AX2THf2"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "UJSNUWUjTLu0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.under_sampling import NearMiss\n",
        "nm = NearMiss()\n",
        "X, Y = nm.fit_resample(X, Y)"
      ],
      "metadata": {
        "id": "Coi8AJXETN36"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ],
      "metadata": {
        "id": "jdod--JpTQxy"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ],
      "metadata": {
        "id": "25GbMtsTTS2q"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann = Sequential()"
      ],
      "metadata": {
        "id": "Zj7qSu75TVPi"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Dense(256, activation = 'relu', input_shape=(X_train.shape[1], 1)))\n",
        "ann.add(Dense(256, activation = 'relu'))\n",
        "ann.add(Dense(128, activation = 'relu'))"
      ],
      "metadata": {
        "id": "NQPS18GDTXlN"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "EDoH7vVSTaKZ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Flatten())"
      ],
      "metadata": {
        "id": "JxhSR850Tc2D"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Dense(64, activation='relu'))\n",
        "ann.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "7c09nxmITfqW"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "RHm8NHK9Thek"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "opNjqgvfTjp2",
        "outputId": "9054af4e-64a5-4eec-df12-8a532e41514d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "17/17 [==============================] - 3s 24ms/step - loss: 0.6665 - accuracy: 0.5213\n",
            "Epoch 2/75\n",
            "17/17 [==============================] - 0s 29ms/step - loss: 0.5449 - accuracy: 0.7258\n",
            "Epoch 3/75\n",
            "17/17 [==============================] - 0s 28ms/step - loss: 0.5080 - accuracy: 0.7326\n",
            "Epoch 4/75\n",
            "17/17 [==============================] - 0s 27ms/step - loss: 0.4472 - accuracy: 0.7674\n",
            "Epoch 5/75\n",
            "17/17 [==============================] - 1s 30ms/step - loss: 0.4294 - accuracy: 0.7762\n",
            "Epoch 6/75\n",
            "17/17 [==============================] - 1s 30ms/step - loss: 0.4085 - accuracy: 0.8062\n",
            "Epoch 7/75\n",
            "17/17 [==============================] - 0s 23ms/step - loss: 0.4069 - accuracy: 0.8014\n",
            "Epoch 8/75\n",
            "17/17 [==============================] - 0s 24ms/step - loss: 0.3872 - accuracy: 0.8072\n",
            "Epoch 9/75\n",
            "17/17 [==============================] - 1s 30ms/step - loss: 0.3737 - accuracy: 0.8275\n",
            "Epoch 10/75\n",
            "17/17 [==============================] - 1s 31ms/step - loss: 0.3636 - accuracy: 0.8275\n",
            "Epoch 11/75\n",
            "17/17 [==============================] - 1s 31ms/step - loss: 0.3638 - accuracy: 0.8401\n",
            "Epoch 12/75\n",
            "17/17 [==============================] - 1s 31ms/step - loss: 0.3516 - accuracy: 0.8537\n",
            "Epoch 13/75\n",
            "17/17 [==============================] - 0s 29ms/step - loss: 0.3541 - accuracy: 0.8508\n",
            "Epoch 14/75\n",
            "17/17 [==============================] - 0s 19ms/step - loss: 0.3415 - accuracy: 0.8527\n",
            "Epoch 15/75\n",
            "17/17 [==============================] - 0s 18ms/step - loss: 0.3311 - accuracy: 0.8643\n",
            "Epoch 16/75\n",
            "17/17 [==============================] - 0s 18ms/step - loss: 0.3351 - accuracy: 0.8740\n",
            "Epoch 17/75\n",
            "17/17 [==============================] - 0s 18ms/step - loss: 0.3302 - accuracy: 0.8692\n",
            "Epoch 18/75\n",
            "17/17 [==============================] - 0s 19ms/step - loss: 0.3201 - accuracy: 0.8721\n",
            "Epoch 19/75\n",
            "17/17 [==============================] - 0s 17ms/step - loss: 0.3177 - accuracy: 0.8876\n",
            "Epoch 20/75\n",
            "17/17 [==============================] - 0s 19ms/step - loss: 0.3211 - accuracy: 0.8828\n",
            "Epoch 21/75\n",
            "17/17 [==============================] - 0s 19ms/step - loss: 0.2900 - accuracy: 0.8944\n",
            "Epoch 22/75\n",
            "17/17 [==============================] - 0s 18ms/step - loss: 0.2788 - accuracy: 0.9050\n",
            "Epoch 23/75\n",
            "17/17 [==============================] - 0s 19ms/step - loss: 0.2769 - accuracy: 0.9002\n",
            "Epoch 24/75\n",
            "17/17 [==============================] - 0s 19ms/step - loss: 0.2605 - accuracy: 0.9031\n",
            "Epoch 25/75\n",
            "17/17 [==============================] - 0s 18ms/step - loss: 0.2650 - accuracy: 0.8992\n",
            "Epoch 26/75\n",
            "17/17 [==============================] - 0s 19ms/step - loss: 0.2621 - accuracy: 0.8953\n",
            "Epoch 27/75\n",
            "17/17 [==============================] - 0s 20ms/step - loss: 0.2586 - accuracy: 0.8992\n",
            "Epoch 28/75\n",
            "17/17 [==============================] - 0s 19ms/step - loss: 0.2575 - accuracy: 0.8934\n",
            "Epoch 29/75\n",
            "17/17 [==============================] - 0s 19ms/step - loss: 0.2471 - accuracy: 0.9089\n",
            "Epoch 30/75\n",
            "17/17 [==============================] - 0s 18ms/step - loss: 0.2305 - accuracy: 0.9176\n",
            "Epoch 31/75\n",
            "17/17 [==============================] - 0s 20ms/step - loss: 0.2342 - accuracy: 0.9196\n",
            "Epoch 32/75\n",
            "17/17 [==============================] - 0s 18ms/step - loss: 0.2135 - accuracy: 0.9273\n",
            "Epoch 33/75\n",
            "17/17 [==============================] - 0s 18ms/step - loss: 0.2461 - accuracy: 0.9109\n",
            "Epoch 34/75\n",
            "17/17 [==============================] - 0s 19ms/step - loss: 0.2139 - accuracy: 0.9128\n",
            "Epoch 35/75\n",
            "17/17 [==============================] - 0s 18ms/step - loss: 0.2127 - accuracy: 0.9273\n",
            "Epoch 36/75\n",
            "17/17 [==============================] - 0s 18ms/step - loss: 0.2248 - accuracy: 0.9070\n",
            "Epoch 37/75\n",
            "17/17 [==============================] - 0s 18ms/step - loss: 0.2353 - accuracy: 0.9041\n",
            "Epoch 38/75\n",
            "17/17 [==============================] - 0s 18ms/step - loss: 0.2125 - accuracy: 0.9273\n",
            "Epoch 39/75\n",
            "17/17 [==============================] - 0s 18ms/step - loss: 0.2042 - accuracy: 0.9225\n",
            "Epoch 40/75\n",
            "17/17 [==============================] - 0s 19ms/step - loss: 0.1845 - accuracy: 0.9380\n",
            "Epoch 41/75\n",
            "17/17 [==============================] - 0s 18ms/step - loss: 0.2152 - accuracy: 0.9157\n",
            "Epoch 42/75\n",
            "17/17 [==============================] - 0s 19ms/step - loss: 0.2154 - accuracy: 0.9186\n",
            "Epoch 43/75\n",
            "17/17 [==============================] - 0s 19ms/step - loss: 0.1717 - accuracy: 0.9428\n",
            "Epoch 44/75\n",
            "17/17 [==============================] - 0s 22ms/step - loss: 0.1917 - accuracy: 0.9273\n",
            "Epoch 45/75\n",
            "17/17 [==============================] - 1s 44ms/step - loss: 0.1598 - accuracy: 0.9486\n",
            "Epoch 46/75\n",
            "17/17 [==============================] - 1s 44ms/step - loss: 0.1446 - accuracy: 0.9583\n",
            "Epoch 47/75\n",
            "17/17 [==============================] - 1s 44ms/step - loss: 0.1435 - accuracy: 0.9545\n",
            "Epoch 48/75\n",
            "17/17 [==============================] - 1s 46ms/step - loss: 0.1353 - accuracy: 0.9564\n",
            "Epoch 49/75\n",
            "17/17 [==============================] - 0s 29ms/step - loss: 0.1228 - accuracy: 0.9603\n",
            "Epoch 50/75\n",
            "17/17 [==============================] - 1s 30ms/step - loss: 0.1219 - accuracy: 0.9622\n",
            "Epoch 51/75\n",
            "17/17 [==============================] - 1s 32ms/step - loss: 0.1339 - accuracy: 0.9545\n",
            "Epoch 52/75\n",
            "17/17 [==============================] - 1s 33ms/step - loss: 0.1261 - accuracy: 0.9574\n",
            "Epoch 53/75\n",
            "17/17 [==============================] - 1s 32ms/step - loss: 0.1192 - accuracy: 0.9603\n",
            "Epoch 54/75\n",
            "17/17 [==============================] - 1s 31ms/step - loss: 0.1098 - accuracy: 0.9738\n",
            "Epoch 55/75\n",
            "17/17 [==============================] - 0s 27ms/step - loss: 0.1115 - accuracy: 0.9651\n",
            "Epoch 56/75\n",
            "17/17 [==============================] - 1s 33ms/step - loss: 0.1045 - accuracy: 0.9690\n",
            "Epoch 57/75\n",
            "17/17 [==============================] - 1s 30ms/step - loss: 0.0890 - accuracy: 0.9777\n",
            "Epoch 58/75\n",
            "17/17 [==============================] - 1s 30ms/step - loss: 0.1151 - accuracy: 0.9574\n",
            "Epoch 59/75\n",
            "17/17 [==============================] - 0s 27ms/step - loss: 0.1541 - accuracy: 0.9419\n",
            "Epoch 60/75\n",
            "17/17 [==============================] - 1s 29ms/step - loss: 0.1412 - accuracy: 0.9477\n",
            "Epoch 61/75\n",
            "17/17 [==============================] - 1s 32ms/step - loss: 0.0911 - accuracy: 0.9709\n",
            "Epoch 62/75\n",
            "17/17 [==============================] - 0s 28ms/step - loss: 0.0953 - accuracy: 0.9671\n",
            "Epoch 63/75\n",
            "17/17 [==============================] - 1s 31ms/step - loss: 0.0728 - accuracy: 0.9787\n",
            "Epoch 64/75\n",
            "17/17 [==============================] - 1s 31ms/step - loss: 0.0731 - accuracy: 0.9806\n",
            "Epoch 65/75\n",
            "17/17 [==============================] - 1s 34ms/step - loss: 0.0753 - accuracy: 0.9767\n",
            "Epoch 66/75\n",
            "17/17 [==============================] - 1s 30ms/step - loss: 0.0712 - accuracy: 0.9816\n",
            "Epoch 67/75\n",
            "17/17 [==============================] - 1s 38ms/step - loss: 0.0731 - accuracy: 0.9777\n",
            "Epoch 68/75\n",
            "17/17 [==============================] - 1s 36ms/step - loss: 0.0692 - accuracy: 0.9787\n",
            "Epoch 69/75\n",
            "17/17 [==============================] - 1s 32ms/step - loss: 0.1146 - accuracy: 0.9641\n",
            "Epoch 70/75\n",
            "17/17 [==============================] - 1s 32ms/step - loss: 0.0919 - accuracy: 0.9709\n",
            "Epoch 71/75\n",
            "17/17 [==============================] - 1s 31ms/step - loss: 0.0710 - accuracy: 0.9797\n",
            "Epoch 72/75\n",
            "17/17 [==============================] - 0s 25ms/step - loss: 0.0651 - accuracy: 0.9835\n",
            "Epoch 73/75\n",
            "17/17 [==============================] - 0s 19ms/step - loss: 0.0888 - accuracy: 0.9758\n",
            "Epoch 74/75\n",
            "17/17 [==============================] - 0s 18ms/step - loss: 0.0883 - accuracy: 0.9719\n",
            "Epoch 75/75\n",
            "17/17 [==============================] - 0s 18ms/step - loss: 0.0856 - accuracy: 0.9690\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x794cc2b475e0>"
            ]
          },
          "metadata": {},
          "execution_count": 217
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = ann.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MTT-pakZTltJ",
        "outputId": "41f41a03-c610-463d-ad87-cef9e53ac134"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "9/9 [==============================] - 0s 5ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "AWbEg8AFTqKT",
        "outputId": "87370d52-bc7f-4f9b-a2d7-7ceb28f512b8"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9108527131782945,\n",
              " 0.984375,\n",
              " 0.8571428571428571,\n",
              " 0.8218980731136323,\n",
              " 0.8309407788540516)"
            ]
          },
          "metadata": {},
          "execution_count": 219
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "A4E8ayYaTtRR",
        "outputId": "2e24dcf4-e592-4c63-f361-cee41996ce41"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.8384615384615385"
            ]
          },
          "metadata": {},
          "execution_count": 220
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **MLP(LSA)**"
      ],
      "metadata": {
        "id": "9913NnfPS-Ux"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "mlp = MLPClassifier(hidden_layer_sizes=(8,7), learning_rate_init=0.1, random_state= 50)"
      ],
      "metadata": {
        "id": "1UxUpQVlTow0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cv = KFold(n_splits=5, random_state=1, shuffle=True)"
      ],
      "metadata": {
        "id": "mfHOP7rgUA-y"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Imbalanced**"
      ],
      "metadata": {
        "id": "_1bDRWaoVjDF"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/LSA_TR.csv')"
      ],
      "metadata": {
        "id": "BTCZnQCqTCOq"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "JZU-DIIxTleY"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cross_val_predict(mlp, X, Y, cv=cv, n_jobs=-1)"
      ],
      "metadata": {
        "id": "N5SnsMbcUQlB"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(Y, pred), recall_score(Y, pred), precision_score(Y, pred), f1_score(Y, pred), cohen_kappa_score(Y, pred), matthews_corrcoef(Y, pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "v2fq8jb6UhWR",
        "outputId": "cf87fcc4-01b9-4b3a-a3d7-ba66e6f6a03b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9079640533002789,\n",
              " 0.8666666666666667,\n",
              " 0.7259740259740259,\n",
              " 0.7901060070671378,\n",
              " 0.7317535730728917,\n",
              " 0.7364621099480211)"
            ]
          },
          "metadata": {},
          "execution_count": 122
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(Y, pred)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MrSLNku-VanZ",
        "outputId": "5dc6d554-2da7-415d-d3d7-29b557548ea4"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9182804027885361"
            ]
          },
          "metadata": {},
          "execution_count": 123
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Test**"
      ],
      "metadata": {
        "id": "Zp88cOba688e"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/LSA_TR.csv')\n",
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "62xvPhEM6_cy"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.3, random_state = 1)"
      ],
      "metadata": {
        "id": "tUk1WzSeoEzB"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "mlp.fit(xtrain, ytrain)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 92
        },
        "id": "0UVwBf_v7Gqc",
        "outputId": "a332a98b-3fa7-4b4f-c605-4cc3b05199a6"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "MLPClassifier(hidden_layer_sizes=(8, 7), learning_rate_init=0.1,\n",
              "              random_state=50)"
            ],
            "text/html": [
              "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>MLPClassifier(hidden_layer_sizes=(8, 7), learning_rate_init=0.1,\n",
              "              random_state=50)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">MLPClassifier</label><div class=\"sk-toggleable__content\"><pre>MLPClassifier(hidden_layer_sizes=(8, 7), learning_rate_init=0.1,\n",
              "              random_state=50)</pre></div></div></div></div></div>"
            ]
          },
          "metadata": {},
          "execution_count": 61
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = mlp.predict(xtest)"
      ],
      "metadata": {
        "id": "OTNWNkd07Izy"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(ytest, pred), recall_score(ytest, pred), precision_score(ytest, pred), f1_score(ytest, pred), cohen_kappa_score(ytest, pred), matthews_corrcoef(ytest, pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "jfaylTLU7K0z",
        "outputId": "37a79724-39c2-45e0-d069-8b47816a65dd"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.8937048503611971,\n",
              " 0.7477477477477478,\n",
              " 0.7793427230046949,\n",
              " 0.7632183908045977,\n",
              " 0.6947266037199145,\n",
              " 0.6949738924098722)"
            ]
          },
          "metadata": {},
          "execution_count": 66
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(ytest, pred)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bzQoQk9Z7Pbk",
        "outputId": "932bb8b4-7bd1-4031-de30-c626fdc29ba5"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9370816599732262"
            ]
          },
          "metadata": {},
          "execution_count": 64
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**ADASYN**"
      ],
      "metadata": {
        "id": "aNcdZak5Vqpz"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/LSA_TR.csv')\n",
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "dyX-grsvVstD"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.over_sampling import ADASYN\n",
        "ada = ADASYN(random_state=42)\n",
        "X, Y = ada.fit_resample(X, Y)"
      ],
      "metadata": {
        "id": "4e6UZmLfV9EC"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cross_val_predict(mlp, X, Y, cv=cv, n_jobs=-1)"
      ],
      "metadata": {
        "id": "h2oyKXu6WBtE"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(Y, pred), recall_score(Y, pred), precision_score(Y, pred), f1_score(Y, pred), cohen_kappa_score(Y, pred), matthews_corrcoef(Y, pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7XiIiXHoWF7r",
        "outputId": "1387fce1-58df-40d1-e78a-8e6a72e0a37e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9512569564383035,\n",
              " 0.9699505515405097,\n",
              " 0.9357798165137615,\n",
              " 0.9525588345162496,\n",
              " 0.902473569156397,\n",
              " 0.9030872845532104)"
            ]
          },
          "metadata": {},
          "execution_count": 127
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(Y, pred)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_7I60WikWIFr",
        "outputId": "f2af9348-e24e-4811-93f1-660d2ed572f4"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9322230828814873"
            ]
          },
          "metadata": {},
          "execution_count": 128
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**SMOTETomek**"
      ],
      "metadata": {
        "id": "47BX96JhWcSa"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/LSA_TR.csv')\n",
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "hgU3jLveWeoB"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.combine import SMOTETomek\n",
        "smt = SMOTETomek(random_state=42)\n",
        "X, Y = smt.fit_resample(X, Y)"
      ],
      "metadata": {
        "id": "NfOD2EzxWh_b"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cross_val_predict(mlp, X, Y, cv=cv, n_jobs=-1)"
      ],
      "metadata": {
        "id": "qUTw7xdMWmFq"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(Y, pred), recall_score(Y, pred), precision_score(Y, pred), f1_score(Y, pred), cohen_kappa_score(Y, pred), matthews_corrcoef(Y, pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "dKj6XGgLWoSa",
        "outputId": "1507abde-f22a-4f6b-c551-05e700468af0"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.94713400464756,\n",
              " 0.9674670797831139,\n",
              " 0.9296613323409006,\n",
              " 0.9481875118618335,\n",
              " 0.8942680092951201,\n",
              " 0.8950083691924998)"
            ]
          },
          "metadata": {},
          "execution_count": 132
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(Y, pred)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "hh0hF6LsWrNo",
        "outputId": "6e37b530-3113-4ede-9f34-3f04124f6393"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9268009295120062"
            ]
          },
          "metadata": {},
          "execution_count": 30
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**NearMiss**"
      ],
      "metadata": {
        "id": "7wSvmowHW1CG"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/LSA_TR.csv')\n",
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "SJ8vMts7W3Oq"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.under_sampling import NearMiss\n",
        "nm = NearMiss()\n",
        "X, Y = nm.fit_resample(X, Y)"
      ],
      "metadata": {
        "id": "bhbpI7tZW8Ty"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cross_val_predict(mlp, X, Y, cv=cv, n_jobs=-1)"
      ],
      "metadata": {
        "id": "-WeLKfqtXB8z"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(Y, pred), recall_score(Y, pred), precision_score(Y, pred), f1_score(Y, pred), cohen_kappa_score(Y, pred), matthews_corrcoef(Y, pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "EK2cpT4bXFmR",
        "outputId": "845ce010-e1b7-46fd-dd7f-78ae6d6dee22"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9015503875968992,\n",
              " 0.9116279069767442,\n",
              " 0.8936170212765957,\n",
              " 0.9025326170376055,\n",
              " 0.8031007751937984,\n",
              " 0.8032639449503535)"
            ]
          },
          "metadata": {},
          "execution_count": 136
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(Y, pred)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "pBW8A8ofXIYC",
        "outputId": "a874711b-f1e6-42d4-c420-c19bd5b9f0e5"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.8914728682170543"
            ]
          },
          "metadata": {},
          "execution_count": 35
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **MLP(NMBroto)**"
      ],
      "metadata": {
        "id": "oEL3quAEXPyh"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Imbalanced**"
      ],
      "metadata": {
        "id": "yLVrLhHBXV85"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/NMB-TR.csv')\n",
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "zk4jxn-JXTU5"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cross_val_predict(mlp, X, Y, cv=cv, n_jobs=-1)"
      ],
      "metadata": {
        "id": "Ic8lk7B5Xb0R"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(Y, pred), recall_score(Y, pred), precision_score(Y, pred), f1_score(Y, pred), cohen_kappa_score(Y, pred), matthews_corrcoef(Y, pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "NOng0rKzXe3p",
        "outputId": "fb262350-c112-4353-b1e6-686b0c48358c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.8326619150914162,\n",
              " 0.4573643410852713,\n",
              " 0.6082474226804123,\n",
              " 0.5221238938053097,\n",
              " 0.4231510232185537,\n",
              " 0.4294705886842782)"
            ]
          },
          "metadata": {},
          "execution_count": 139
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(Y, pred)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "WnV-XjddXhfk",
        "outputId": "af0614bf-65a3-41b6-9123-3879fcb58dc0"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9264136328427576"
            ]
          },
          "metadata": {},
          "execution_count": 140
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Test**"
      ],
      "metadata": {
        "id": "_G1sOLJcE4nV"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/NMB-TR.csv')\n",
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "75XCAmM6E6GR"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.3, random_state = 1)"
      ],
      "metadata": {
        "id": "65_xjcpdoPRq"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "mlp.fit(xtrain, ytrain)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 92
        },
        "id": "ROQvqJ9hFJl4",
        "outputId": "85d4bffa-ef6d-4a45-b067-95c542a67264"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "MLPClassifier(hidden_layer_sizes=(8, 7), learning_rate_init=0.1,\n",
              "              random_state=50)"
            ],
            "text/html": [
              "<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-2\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>MLPClassifier(hidden_layer_sizes=(8, 7), learning_rate_init=0.1,\n",
              "              random_state=50)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-2\" type=\"checkbox\" checked><label for=\"sk-estimator-id-2\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">MLPClassifier</label><div class=\"sk-toggleable__content\"><pre>MLPClassifier(hidden_layer_sizes=(8, 7), learning_rate_init=0.1,\n",
              "              random_state=50)</pre></div></div></div></div></div>"
            ]
          },
          "metadata": {},
          "execution_count": 112
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = mlp.predict(xtest)"
      ],
      "metadata": {
        "id": "5lgKRJZsFLm5"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(ytest, pred), recall_score(ytest, pred), precision_score(ytest, pred), f1_score(ytest, pred), cohen_kappa_score(ytest, pred), matthews_corrcoef(ytest, pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "flEqq4LhFNji",
        "outputId": "545d5c9b-4867-46ab-c27a-f2fa05c6c667"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.8163054695562435,\n",
              " 0.6036036036036037,\n",
              " 0.5982142857142857,\n",
              " 0.6008968609865472,\n",
              " 0.48159702811389826,\n",
              " 0.481605198071554)"
            ]
          },
          "metadata": {},
          "execution_count": 114
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(ytest, pred)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "u79VEya2FRLx",
        "outputId": "78886b41-cba2-4f8f-c0ff-46498f8fab35"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.8795180722891566"
            ]
          },
          "metadata": {},
          "execution_count": 115
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**ADASYN**"
      ],
      "metadata": {
        "id": "lK748JRxXkyq"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/NMB-TR.csv')\n",
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "Ta5NG8i-Xnv6"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.over_sampling import ADASYN\n",
        "ada = ADASYN(random_state=42)\n",
        "X, Y = ada.fit_resample(X, Y)"
      ],
      "metadata": {
        "id": "nzZI_M_CXtKC"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cross_val_predict(mlp, X, Y, cv=cv, n_jobs=-1)"
      ],
      "metadata": {
        "id": "mdTcnAXcXwWi"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(Y, pred), recall_score(Y, pred), precision_score(Y, pred), f1_score(Y, pred), cohen_kappa_score(Y, pred), matthews_corrcoef(Y, pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "u5TfipYbXyxj",
        "outputId": "ab42f428-636d-409c-ac3c-51cf5fc42097"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.8128933816517261,\n",
              " 0.8959037955655769,\n",
              " 0.772020725388601,\n",
              " 0.8293616281092364,\n",
              " 0.6247806793728854,\n",
              " 0.6332838726749025)"
            ]
          },
          "metadata": {},
          "execution_count": 145
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(Y, pred)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "KKOHNgxSX1tZ",
        "outputId": "e6333753-58ff-4b98-e62c-20d91dc455d0"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.7273431448489543"
            ]
          },
          "metadata": {},
          "execution_count": 146
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**SMOTETomek**"
      ],
      "metadata": {
        "id": "Hb0fePqBX_qw"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/NMB-TR.csv')\n",
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "sAu4Z5RsYCbS"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.combine import SMOTETomek\n",
        "smt = SMOTETomek(random_state=42)\n",
        "X, Y = smt.fit_resample(X, Y)"
      ],
      "metadata": {
        "id": "VNZc923hYE36"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cross_val_predict(mlp, X, Y, cv=cv, n_jobs=-1)"
      ],
      "metadata": {
        "id": "qCz5_wFAYJnz"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(Y, pred), recall_score(Y, pred), precision_score(Y, pred), f1_score(Y, pred), cohen_kappa_score(Y, pred), matthews_corrcoef(Y, pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ywcgbGs9YMAq",
        "outputId": "f71b1132-29b2-4024-8f67-81f6a8832310"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.8255228505034856,\n",
              " 0.8415956622773044,\n",
              " 0.8153846153846154,\n",
              " 0.8282828282828283,\n",
              " 0.6510457010069713,\n",
              " 0.6513823380698701)"
            ]
          },
          "metadata": {},
          "execution_count": 150
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(Y, pred)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "dzk7GMdMYOlB",
        "outputId": "9ebfa1aa-a990-446e-b829-92a8a262de4c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.8094500387296669"
            ]
          },
          "metadata": {},
          "execution_count": 151
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**NearMiss**"
      ],
      "metadata": {
        "id": "xTOk0BJYYRQJ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/NMB-TR.csv')\n",
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "d7FSP4OfYTei"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.under_sampling import NearMiss\n",
        "nm = NearMiss()\n",
        "X, Y = nm.fit_resample(X, Y)"
      ],
      "metadata": {
        "id": "H5l1oRQMYWMq"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cross_val_predict(mlp, X, Y, cv=cv, n_jobs=-1)"
      ],
      "metadata": {
        "id": "B8muMiPKYaTR"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(Y, pred), recall_score(Y, pred), precision_score(Y, pred), f1_score(Y, pred), cohen_kappa_score(Y, pred), matthews_corrcoef(Y, pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "0ZcWbhcvYcxK",
        "outputId": "39f5e3de-359c-42a6-e6e8-2a6d57a16471"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.8565891472868217,\n",
              " 0.862015503875969,\n",
              " 0.852760736196319,\n",
              " 0.8573631457208944,\n",
              " 0.7131782945736433,\n",
              " 0.7132202978471306)"
            ]
          },
          "metadata": {},
          "execution_count": 155
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(Y, pred)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "kmLgK9r1YfO6",
        "outputId": "6e759d91-526f-40a7-cb61-cbdf4d5a3f86"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.8511627906976744"
            ]
          },
          "metadata": {},
          "execution_count": 156
        }
      ]
    }
  ],
  "metadata": {
    "colab": {
      "collapsed_sections": [
        "DdbnMkhvbzYC",
        "8CkwkgKyMvRS",
        "oVYoQE_LiUc7",
        "o-o-yNFvzoKQ",
        "vOo7m3UdDDdE",
        "bZQ5oWq7JLDW",
        "RzCkZcczQedc",
        "oEL3quAEXPyh"
      ],
      "provenance": []
    },
    "gpuClass": "standard",
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}

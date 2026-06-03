# convert_h5_to_pkl.py
import tensorflow as tf
import joblib

def convert():
    model_path = 'direct_reward_model.h5'
    save_path = 'direct_reward_weights.pkl'
    
    print(f"{model_path} を読み込んでいます...")
    model = tf.keras.models.load_model(model_path)
    
    weights = []
    biases = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            w, b = layer.get_weights()
            weights.append(w)
            biases.append(b)
            
    # NumPy配列のリストとして保存
    joblib.dump({'weights': weights, 'biases': biases}, save_path)
    print(f"重みの抽出が完了しました。{save_path} に保存しました！")

if __name__ == "__main__":
    convert()
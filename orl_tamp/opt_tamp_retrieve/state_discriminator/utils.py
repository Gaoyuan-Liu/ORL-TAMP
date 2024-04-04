
import pickle

def save_list(list, filename):
    with open(filename+'.pkl', 'wb') as f:
        pickle.dump(list, f)

def load_list(filename):
    with open(filename+'.pkl', 'rb') as f:
        list = pickle.load(f)
    return list



    

if __name__ == '__main__':
    main()
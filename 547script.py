LIMIT = 547

def isLimit(map):
    for key in map:
        if map[key] < LIMIT:
            return False
    return True

def saveData(data):
    with open('face-emo.csv', 'w') as file:
        file.writelines(data)
    print("data saved under face-emo.csv")
    return 


""" 
    { 0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprize", 6: "Neutral" }
    
    This function is used to balance the data between emotions so we have an
    even number of pixels for each categories. The new data is saved in a new
    file called face-emo.csv.
    
    TODO: We might need to remove the Usage Column
"""
def fetch_data():
    data = []
    classes = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

    with open('fer2013.csv', 'r') as file:
        data.append(file.readline())

        for line in file:
            emo = int(line.split(',')[0])
            
            if classes[emo] != LIMIT:
                data.append(line)
                classes[emo] += 1
        
            if isLimit(classes) == True:
                break

        print(classes)

        # print(f"length of data should equal 547 * 7. data = {len(data) - 1} == {547 * 7}")
        # Saving new data
        saveData(data)

    return 


if __name__ == "__main__":
    fetch_data()
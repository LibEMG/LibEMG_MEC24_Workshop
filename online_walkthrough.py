import libemg
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def main():
    # -------------------------------------------------------------------------------------
    # - The first thing we need to do is establish a streamer. Streamers are utilities    -
    # - within LibEMG that are hardware-specific, and are responsible for grabbing the    -
    # - data using whatever communication library is required (serial, bluetooth, etc.)   -
    # - and stores this data in hardware-agnostic, shared-memory format that can be       -
    # - hooked into various other LibEMG components, or your even your own code.          - 
    # -------------------------------------------------------------------------------------

    # For a delsys streamer, you can run the below line:
    # The delsys streamer can be configured to work with any delsys base station, and any number of 
    # supported sensors.
    # license = ""
    # key     = ""
    # delsys_process, shared_memory_items = libemg.streamers.delsys_api_streamer(license=license,
    #                                                                            key = key,
    #                                                                            num_channels=4)
    
    # Altenatively, for a sifi streamer, you can run the below line:
    # The sifi streamer can be configured to use the biopoint or bioarmband, and include the
    # full multi-modal suite (EMG, ECG, EDA, IMU, PPG).
    mac = "CA:C0:CE:88:87:C1"
    sifi_process, shared_memory_items = libemg.streamers.sifibridge_streamer(mac=mac)

    # The delsys_process and sifi_process are handles to the process that runs in the background.
    # The shared memory items is a list of descriptors for where the data is stored, and other
    # LibEMG utilities hook into this 


    # -------------------------------------------------------------------------------------
    # - Once a streamer has started to run, we can use an OnlineDataHandler, another      -
    # - LibEMG utility, to hook into that shared memory location.                         -
    # - To do this, we need to supply the shared memory items returned by the streamer.   -
    # -------------------------------------------------------------------------------------
    
    online_data_handler = libemg.data_handler.OnlineDataHandler(shared_memory_items=shared_memory_items)

    # We can peek into what the streamer has collected thus far with the .get_data method
    time.sleep(10)
    data, counts = online_data_handler.get_data()
    print(data)
    print(counts)
    # If we had a multi-modal streamer, each modality would have its own key in this dictionary.

    # The online_data_handler is used by other parts of the online pipeline, like the 
    # OnlineEMGClassifier, but has some of its own methods that allow us to visualize the
    # streaming data, or log it to a file.

    # By specifying block=True, we tell the code to keep visualizing the signal until we 
    # close the window.
    online_data_handler.visualize(num_samples=5000, block=True)

    # Logging is commonly done in the background, so we can specify block=False.
    online_data_handler.log_to_file(block=False)

    # Even though this is being performed in another process, we still have control over 
    # the logging or plotting process, and can stop background operations whenever we want.
    time.sleep(10)
    online_data_handler.stop_log()



    # -------------------------------------------------------------------------------------
    # - Now we can move on to other components that make use of the online_data_handler,  -
    # - such as screen guided training.                                                   -
    # -------------------------------------------------------------------------------------
    gui = libemg.gui.GUI(online_data_handler=online_data_handler)
    
    # To use the screen guided training, we need pictures of the prompts we want to use.
    # We can reach out to another github repository LibEMG hosts to facilitate this
    # (or alternatively use your own images).
    gui.download_gestures(gesture_ids=[1,2,3,4,5],folder='images/')

    # And now we can run the gui
    gui.start_gui()

    # Eventually, we hope to integrate most of LibEMG's capabilities into this GUI for 
    # greater ease of use for clinicians.

    # -------------------------------------------------------------------------------------
    # - With the data we've collected, we can now construct a classifier that runs on     -
    # - live data. To accomplish this, we first need to create an EMGClassifier (like     -
    # - Christian showed us how to do in walkthrough.ipynb                                -
    # -------------------------------------------------------------------------------------

    # First, we gather the data we just collected into an OfflineDataHandler
    offline_data_handler = libemg.data_handler.OfflineDataHandler()
    offline_data_handler.get_data(folder_location="data/",
                                  regex_filters=[libemg.data_handler.RegexFilter("data/C_","_R_",["0","1","2","3","4"],"classes"),
                                                 libemg.data_handler.RegexFilter("_R_","_emg.csv",["0","1","2","3","4"],"reps")])
    
    # Now we need to decide on windowing parameters and a feature set
    window_size       = 196
    window_increment  = 56
    features          = ["MFL","RMS","ZS","SSC"]

    # perform windowing
    windows, metadata = offline_data_handler.parse_windows(window_size=window_size, window_increment=window_increment)

    # extract and verify the features look okay
    fe                = libemg.feature_extractor.FeatureExtractor()
    features          = fe.extract_features(features, windows)
    fe.visualize_feature_space(feature_dic=features, projection="PCA", classes=metadata["classes"], savedir="./",render=False)

    # Let's build the classifier!
    offline_classifier = libemg.emg_predictor.EMGClassifier("LDA")
    feature_set        = {
                        'training_features': features,
                        'training_labels': metadata["classes"]
    }
    offline_classifier.fit(feature_set)

    # -------------------------------------------------------------------------------------
    # - Making the OnlineEMGClassifier just involves constructing the pipeline with the   -
    # - components we're made thus far (online_data_handler, offline_classifier, and      -
    # - some descriptors like window size, increment, and features).                      -
    # -------------------------------------------------------------------------------------
    online_classifier = libemg.emg_predictor.OnlineEMGClassifier(offline_classifier=offline_classifier,
                                                                 window_size=window_size,
                                                                 window_increment=window_increment,
                                                                 online_data_handler=online_data_handler,
                                                                 features = features,
                                                                 std_out=True)
    # And now we just run the classifier!
    online_classifier.run(block=True)

if __name__ == "__main__":
    # -------------------------------------------------------------------------------------
    # - This is the entry point for your script.                                          -
    # - An important part of the online LibEMG workflow are the background processes that -
    # - are spawned to run the streamer, visualizers, and online classifier. Background   -
    # - processes are only able to spawn when you use if __name__ == "__main__".          -
    # -------------------------------------------------------------------------------------
    main()
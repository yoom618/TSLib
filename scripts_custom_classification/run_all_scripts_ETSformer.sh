# UEA_MTSC30=("EigenWorms") # GPU 3
# UEA_MTSC30=("EthanolConcentration")  # GPU 2
# UEA_MTSC30=("MotorImagery") # GPU 0
# UEA_MTSC30=("ArticularyWordRecognition" "AtrialFibrillation" "BasicMotions" "CharacterTrajectories")
# UEA_MTSC30=("Cricket" "DuckDuckGeese" "Epilepsy" "ERing" "FaceDetection")
# UEA_MTSC30=("FingerMovements" "HandMovementDirection" "Handwriting" "Heartbeat" "InsectWingbeat")
# UEA_MTSC30=("JapaneseVowels" "Libras" "LSST" "NATOPS" "PEMS-SF" )
# UEA_MTSC30=("PenDigits" "PhonemeSpectra" "RacketSports")
# UEA_MTSC30=("UWaveGestureLibrary" "SelfRegulationSCP1" "SelfRegulationSCP2" "SpokenArabicDigits" "StandWalkJump")

UEA_MTSC30=("EigenWorms(b2b4)")
model="ETSformer_CLS"
for dataset in ${UEA_MTSC30[@]}
do
    echo "Running ./scripts_custom_classification/scripts_baseline/${model}_${dataset}.sh"
    echo "Result will be saved in ./scripts_custom_classification/results/${model}_${dataset}.out"
    nohup bash ./scripts_custom_classification/scripts_baseline/${model}_${dataset}.sh > ./scripts_custom_classification/results/${model}_${dataset}.out &
done

UEA_MTSC30=("InsectWingbeat")
model="Crossformer_CLS"
for dataset in ${UEA_MTSC30[@]}
do
    echo "Running ./scripts_custom_classification/scripts_baseline/${model}_${dataset}.sh"
    echo "Result will be saved in ./scripts_custom_classification/results/${model}_${dataset}.out"
    nohup bash ./scripts_custom_classification/scripts_baseline/${model}_${dataset}.sh > ./scripts_custom_classification/results/${model}_${dataset}.out &
done


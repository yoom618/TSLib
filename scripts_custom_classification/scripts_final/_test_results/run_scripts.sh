UEA_MTSC30=("ArticularyWordRecognition" "AtrialFibrillation" "BasicMotions" "CharacterTrajectories" "Cricket" \
            "DuckDuckGeese" "EigenWorms" "Epilepsy" "ERing" "EthanolConcentration" \
            "FaceDetection" "FingerMovements" "HandMovementDirection" "Handwriting" "Heartbeat" \
            "InsectWingbeat" "JapaneseVowels" "Libras" "LSST" "MotorImagery" \
            "NATOPS" "PEMS-SF" "PenDigits" "PhonemeSpectra" "RacketSports" \
            "SelfRegulationSCP1" "SelfRegulationSCP2" "SpokenArabicDigits" "StandWalkJump" "UWaveGestureLibrary")


UEA_MTSC30=("All_UEA30")
model="DLinear"
for dataset in ${UEA_MTSC30[@]}
do
    sh_fname="./scripts_custom_classification/scripts_final/${model}/${dataset}.sh"
    out_fname="./scripts_custom_classification/scripts_final/_test_results/${model}_${dataset}.out"
    echo "Running ${sh_fname}"
    echo "Result will be saved in ${out_fname}"
    nohup bash ${sh_fname} > ${out_fname} &
done

pipeline {
    agent any

    environment {
        PROJECT_ID = 'road-for-terraform'
        REGION = 'us-east1'
        REPO_NAME = 'mlops-trading'
        IMAGE_NAME = 'ml-inference'
        CLUSTER_NAME = 'dev-instance'
        MODEL_PATH_1 = 'models/eth_price_predictor.pkl'
        MODEL_PATH_2 = 'models/ethusdt_price_predictor.pkl'
        GCP_SA = credentials('gcp-service-account')
    }

    stages {

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Configurar GCP') {
            steps {
                sh '''
                echo "$GCP_SA" > ${WORKSPACE}/gcp-key.json
                gcloud auth activate-service-account --key-file=${WORKSPACE}/gcp-key.json
                gcloud config set project $PROJECT_ID
                gcloud auth configure-docker $REGION-docker.pkg.dev -q
                '''
            }
        }

        stage('Pré-processamento dos Dados') {
            steps {
                sh '''
                python3 -m venv venv
                source venv/bin/activate
                pip install -r requirements.txt
                python src/preprocess_eth.py
                python src/preprocess_ethusdt.py
                '''
            }
        }

        stage('Treinamento dos Modelos') {
            steps {
                sh '''
                source venv/bin/activate
                python src/train_eth.py --output $MODEL_PATH_1
                python src/train_ethusdt.py --output $MODEL_PATH_2
                '''
            }
        }

        stage('Avaliação dos Modelos') {
            steps {
                sh '''
                source venv/bin/activate
                python src/evaluate.py --model $MODEL_PATH_1 --threshold 0.8
                python src/evaluate.py --model $MODEL_PATH_2 --threshold 0.8
                '''
            }
        }

        stage('Versionamento dos Modelos') {
            steps {
                sh '''
                gsutil cp $MODEL_PATH_1 gs://$PROJECT_ID-model-registry/eth-model-$(date +%Y%m%d%H%M).pkl
                gsutil cp $MODEL_PATH_2 gs://$PROJECT_ID-model-registry/ethusdt-model-$(date +%Y%m%d%H%M).pkl
                '''
            }
        }

        stage('Build e Push da Imagem de Inferência') {
            steps {
                sh '''
                docker build -t $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:$BUILD_NUMBER .
                docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:$BUILD_NUMBER
                '''
            }
        }

        stage('Deploy no GKE') {
            steps {
                sh '''
                gcloud container clusters get-credentials $CLUSTER_NAME --region $REGION --project $PROJECT_ID
                kubectl set image deployment/ml-inference \
                  ml-inference=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:$BUILD_NUMBER \
                  -n ml-inference
                kubectl rollout status deployment/ml-inference -n ml-inference
                '''
            }
        }
    }

    post {
        always {
            cleanWs()
        }
    }
}

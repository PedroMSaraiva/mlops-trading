pipeline {
    agent {
        kubernetes {
            label 'jenkins-agent'
            defaultContainer 'jnlp'
            yamlFile 'k8s/pod-template.yml'
        }
    }

    environment {
        PROJECT_ID = 'road-for-terraform'
        REGION = 'us-east1'
        REPO_NAME = 'mlops-trading'
        IMAGE_NAME = 'ml-inference'
        CLUSTER_NAME = 'dev-instance'
        MODEL_PATH_1 = 'models/eth_price_predictor.pkl'
        MODEL_PATH_2 = 'models/ethusdt_price_predictor.pkl'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Configurar GCP') {
            steps {
                container('gcp-tools') {
                    sh '''
                    # Com Workload Identity não precisa de auth!
                    gcloud config set project $PROJECT_ID
                    gcloud auth configure-docker $REGION-docker.pkg.dev -q
                    '''
                }
            }
        }

        stage('Instalar Dependências Python') {
            steps {
                sh '''
                python3 -m venv venv
                . venv/bin/activate
                pip install --upgrade pip
                pip install -e .
                '''
            }
        }

        stage('Pré-processamento dos Dados') {
            steps {
                sh '''
                . venv/bin/activate
                python src/preprocess_eth.py
                python src/preprocess_ethusdt.py
                '''
            }
        }

        stage('Treinamento dos Modelos') {
            steps {
                sh '''
                . venv/bin/activate
                python src/train_eth.py --output $MODEL_PATH_1
                python src/train_ethusdt.py --output $MODEL_PATH_2
                '''
            }
        }

        stage('Avaliação dos Modelos') {
            steps {
                sh '''
                . venv/bin/activate
                python src/evaluate.py --model $MODEL_PATH_1 --threshold 0.8
                python src/evaluate.py --model $MODEL_PATH_2 --threshold 0.8
                '''
            }
        }

        stage('Versionamento dos Modelos') {
            steps {
                container('gcp-tools') {
                    sh '''
                    gsutil cp $MODEL_PATH_1 gs://$PROJECT_ID-model-registry/eth-model-$(date +%Y%m%d%H%M).pkl
                    gsutil cp $MODEL_PATH_2 gs://$PROJECT_ID-model-registry/ethusdt-model-$(date +%Y%m%d%H%M).pkl
                    '''
                }
            }
        }

        stage('Build e Push da Imagem') {
            steps {
                container('gcp-tools') {
                    sh '''
                    gcloud builds submit \
                      --tag=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:$BUILD_NUMBER \
                      --project=$PROJECT_ID \
                      .
                    '''
                }
            }
        }

        stage('Deploy no GKE') {
            steps {
                container('gcp-tools') {
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
    }

    post {
        always {
            script {
                // Limpar workspace de forma segura com Kubernetes
                try {
                    sh 'rm -rf *'
                } catch (Exception e) {
                    echo "Erro ao limpar workspace: ${e.message}"
                }
            }
        }
    }
}

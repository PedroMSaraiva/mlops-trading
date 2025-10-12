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
                container('python') {
                    sh '''
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install -e .
                    '''
                }
            }
        }

        stage('Preparar e Treinar Modelos') {
            steps {
                container('python') {
                    sh '''
                    . venv/bin/activate
                    # Define variável de ambiente para modo CI/CD
                    export CI=true
                    # Executa o script principal que faz o pré-processamento e treinamento
                    # Em modo CI, usará dados históricos salvos em data/
                    python main.py
                    '''
                }
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
                    # Conectar ao cluster GKE
                    gcloud container clusters get-credentials $CLUSTER_NAME --region $REGION --project $PROJECT_ID
                    
                    # Criar namespace se não existir
                    kubectl apply -f k8s/python-namespace.yml
                    
                    # Atualizar a imagem no deployment
                    export IMAGE_TAG=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:$BUILD_NUMBER
                    sed -i "s|REGISTRY/REPO/ml-inference:latest|$IMAGE_TAG|g" k8s/python-deployment.yml
                    
                    # Aplicar os manifestos do Kubernetes
                    kubectl apply -f k8s/python-deployment.yml
                    kubectl apply -f k8s/python-service.yml
                    
                    # Aguardar o rollout ser concluído
                    kubectl rollout status deployment/ml-inference -n ml-inference --timeout=5m
                    
                    # Mostrar status do deployment
                    kubectl get pods -n ml-inference
                    kubectl get svc -n ml-inference
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
                    // Usa cleanWs() do Jenkins que é mais adequado para ambientes Kubernetes
                    cleanWs(
                        deleteDirs: true,
                        disableDeferredWipeout: true,
                        notFailBuild: true
                    )
                } catch (Exception e) {
                    echo "Erro ao limpar workspace: ${e.message}"
                }
            }
        }
    }
}
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
        REPO_NAME = 'mlops-repo'
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

                    export CI=true

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
                    set +e
                    gcloud builds submit \
                      --tag=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:$BUILD_NUMBER \
                      --project=$PROJECT_ID \
                      . 2>&1 | tee build_output.txt
                    
                    # Extrair Build ID usando sed
                    BUILD_ID=$(grep -i "ID:" build_output.txt | head -1 | sed 's/.*ID: *//i' | awk '{print $1}')
                    
                    if [ -z "$BUILD_ID" ]; then
                      echo "Não foi possível obter o Build ID, mas verificando imagem..."
                      sleep 30
                      if gcloud artifacts docker images describe $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:$BUILD_NUMBER --project=$PROJECT_ID 2>/dev/null; then
                        echo "Imagem criada com sucesso!"
                        exit 0
                      else
                        echo "Falha ao criar imagem"
                        exit 1
                      fi
                    fi
                    
                    echo "Verificando status do build: $BUILD_ID"
                    STATUS=$(gcloud builds describe $BUILD_ID --project=$PROJECT_ID --format="value(status)" 2>/dev/null || echo "UNKNOWN")
                    
                    while [ "$STATUS" = "WORKING" ] || [ "$STATUS" = "QUEUED" ]; do
                      echo "Build ainda em andamento... aguardando 10s"
                      sleep 10
                      STATUS=$(gcloud builds describe $BUILD_ID --project=$PROJECT_ID --format="value(status)")
                    done
                    
                    echo "Build Status: $STATUS"
                    
                    if [ "$STATUS" = "SUCCESS" ]; then
                      echo "Build concluído com sucesso!"
                      exit 0
                    else
                      echo "Build falhou com status: $STATUS"
                      exit 1
                    fi
                    '''
                }
            }
        }

        stage('Deploy no GKE') {
            steps {
                container('gcp-tools') {
                    sh '''
                    if ! command -v kubectl &> /dev/null; then
                        echo "Instalando kubectl..."
                        gcloud components install kubectl --quiet
                    fi
                    
                    if ! command -v gke-gcloud-auth-plugin &> /dev/null; then
                        echo "Instalando gke-gcloud-auth-plugin..."
                        gcloud components install gke-gcloud-auth-plugin --quiet
                    fi
                    
                    export USE_GKE_GCLOUD_AUTH_PLUGIN=True
                    
                    gcloud container clusters get-credentials $CLUSTER_NAME --region $REGION --project $PROJECT_ID
                    
                    kubectl apply -f k8s/python-namespace.yml
                    
                    export IMAGE_TAG=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:$BUILD_NUMBER
                    sed -i "s|REGISTRY/REPO/ml-inference:latest|$IMAGE_TAG|g" k8s/python-deployment.yml
                    
                    kubectl apply -f k8s/python-deployment.yml
                    kubectl apply -f k8s/python-service.yml
                    
                    kubectl rollout status deployment/ml-inference -n ml-inference --timeout=5m
                    
                    echo "=== Pods ==="
                    kubectl get pods -n ml-inference
                    echo "=== Services ==="
                    kubectl get svc -n ml-inference
                    '''
                }
            }
        }
    }

    post {
        always {
            script {
                try {
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
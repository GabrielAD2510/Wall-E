const int IN1 = 18;
const int IN2 = 19;
const int ENA = 21;

void setup() {
  Serial.begin(9600);  // Asegúrate que sea la misma velocidad que en Python
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENA, OUTPUT);

  // Dirección fija del motor (como en tu ejemplo)
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);

  Serial.println("🔧 ESP listo para recibir comandos.");
}

void loop() {
  if (Serial.available() > 0) {
    char comando = Serial.read();

    if (comando == '1') {
      digitalWrite(ENA, HIGH);  // Encender al 100%
      Serial.println("➡️ Motor encendido (derecha).");
    } else if (comando == '0') {
      digitalWrite(ENA, LOW);   // Apagar
      Serial.println("⬅️ Motor apagado (izquierda).");
    } else {
      Serial.print("⚠️ Comando desconocido: ");
      Serial.println(comando);
    }
  }
}

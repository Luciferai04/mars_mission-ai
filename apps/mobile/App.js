import React, { useEffect, useState } from 'react';
import { SafeAreaView, Text, View, StyleSheet } from 'react-native';

export default function App() {
  const [telemetry, setTelemetry] = useState(null);
  const [status, setStatus] = useState('connecting');

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8004/ws/telemetry');
    ws.onopen = () => setStatus('connected');
    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);
        if (msg.type === 'telemetry') setTelemetry(msg.data);
      } catch {}
    };
    ws.onerror = () => setStatus('error');
    ws.onclose = () => setStatus('closed');
    return () => ws.close();
  }, []);

  return (
    <SafeAreaView style={styles.container}>
      <Text style={styles.title}>Mars Mission Mobile</Text>
      <Text style={styles.subtitle}>WebSocket: {status}</Text>
      {telemetry ? (
        <View style={styles.card}>
          <Text>Battery: {telemetry.battery_soc ?? 'n/a'}%</Text>
          <Text>Generation: {telemetry.power_generation ?? 'n/a'} W</Text>
          <Text>Consumption: {telemetry.power_consumption ?? 'n/a'} W</Text>
          <Text>Temp: {telemetry.temp_c ?? 'n/a'} C</Text>
          <Text>Dust: {telemetry.dust_opacity ?? 'n/a'}</Text>
        </View>
      ) : (
        <Text>No telemetry yet</Text>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, alignItems: 'center', justifyContent: 'flex-start', paddingTop: 60 },
  title: { fontSize: 24, fontWeight: 'bold' },
  subtitle: { fontSize: 14, marginBottom: 16 },
  card: { padding: 16, backgroundColor: '#222', borderRadius: 8, width: '90%' },
});

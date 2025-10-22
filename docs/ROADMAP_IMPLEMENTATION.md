# Roadmap Features Implementation (v2.0 → v3.0)

This repository now includes scaffolding and initial implementations for the requested roadmap:

Version 2.0 (Q2 2024)
- Deep Q-Networks (DQN) for MARL: optional DQN backend for all agents
- Multi-rover coordination: fleet coordinator and MARL service endpoint
- Predictive maintenance AI: sklearn-based risk model via Data Integration service
- Enhanced vision with object tracking: optical-flow tracker endpoint

Version 2.1 (Q3 2024)
- Real-time streaming data integration: WebSocket telemetry publisher/subscriber
- Mobile app for mission monitoring: Expo-based app in apps/mobile
- 3D terrain visualization: separate service serving a Three.js viewer
- Voice command interface: voice service mapping commands to planning actions

Version 3.0 (Q4 2024)
- Federated learning across rovers: simple FedAvg aggregator endpoints
- Autonomous experiment design: heuristic experiment proposer
- Long-term strategic planning: strategic planner producing multi-week objectives
- Integration with JPL planning tools: simple exporters (APGEN / PLEXIL-like)

See service and module docs in code headers for usage. Compose services are defined in docker-compose.yml.

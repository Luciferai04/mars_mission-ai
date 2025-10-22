-- Mars Mission Planning Assistant Database Schema
-- PostgreSQL 15+

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search

-- Mission Plans table
CREATE TABLE IF NOT EXISTS mission_plans (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    goals JSONB NOT NULL,
    activities JSONB NOT NULL,
    constraints JSONB,
    power_summary JSONB,
    validation_result JSONB,
    status VARCHAR(50) DEFAULT 'draft',
    sol INT,
    user_id VARCHAR(100),
    version INT DEFAULT 1,
    CONSTRAINT status_check CHECK (status IN ('draft', 'approved', 'executing', 'completed', 'failed'))
);

CREATE INDEX idx_mission_plans_created_at ON mission_plans(created_at DESC);
CREATE INDEX idx_mission_plans_status ON mission_plans(status);
CREATE INDEX idx_mission_plans_sol ON mission_plans(sol);

-- MARL Training Episodes
CREATE TABLE IF NOT EXISTS marl_episodes (
    id SERIAL PRIMARY KEY,
    episode_num INT NOT NULL,
    agent_name VARCHAR(50) NOT NULL,
    reward FLOAT NOT NULL,
    epsilon FLOAT,
    agent_stats JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    training_session VARCHAR(100)
);

CREATE INDEX idx_marl_episodes_session ON marl_episodes(training_session);
CREATE INDEX idx_marl_episodes_num ON marl_episodes(episode_num);

-- Vision Model Results
CREATE TABLE IF NOT EXISTS vision_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    image_path VARCHAR(255) NOT NULL,
    image_hash VARCHAR(64),
    classification VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    processing_time_ms INT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT classification_check CHECK (classification IN ('SAFE', 'CAUTION', 'HAZARD'))
);

CREATE INDEX idx_vision_results_image_hash ON vision_results(image_hash);
CREATE INDEX idx_vision_results_created_at ON vision_results(created_at DESC);
CREATE INDEX idx_vision_results_model_version ON vision_results(model_version);

-- Audit Logs
CREATE TABLE IF NOT EXISTS audit_logs (
    id SERIAL PRIMARY KEY,
    service VARCHAR(50) NOT NULL,
    action VARCHAR(100) NOT NULL,
    user_id VARCHAR(100),
    resource_type VARCHAR(50),
    resource_id VARCHAR(100),
    details JSONB,
    ip_address INET,
    created_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT service_check CHECK (service IN ('planning', 'vision', 'marl', 'data', 'gateway'))
);

CREATE INDEX idx_audit_logs_service ON audit_logs(service);
CREATE INDEX idx_audit_logs_action ON audit_logs(action);
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at DESC);
CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id);

-- NASA Data Cache
CREATE TABLE IF NOT EXISTS nasa_data_cache (
    id SERIAL PRIMARY KEY,
    data_type VARCHAR(50) NOT NULL,
    cache_key VARCHAR(255) NOT NULL UNIQUE,
    data JSONB NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    hit_count INT DEFAULT 0
);

CREATE INDEX idx_nasa_data_cache_key ON nasa_data_cache(cache_key);
CREATE INDEX idx_nasa_data_cache_expires ON nasa_data_cache(expires_at);
CREATE INDEX idx_nasa_data_cache_type ON nasa_data_cache(data_type);

-- DEM File Registry
CREATE TABLE IF NOT EXISTS dem_files (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    storage_path VARCHAR(500) NOT NULL,
    file_size_bytes BIGINT,
    region VARCHAR(100),
    resolution_m FLOAT,
    bounds JSONB,  -- {lat_min, lat_max, lon_min, lon_max}
    uploaded_by VARCHAR(100),
    uploaded_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB
);

CREATE INDEX idx_dem_files_region ON dem_files(region);
CREATE INDEX idx_dem_files_filename ON dem_files(filename);

-- Service Health Status
CREATE TABLE IF NOT EXISTS service_health (
    id SERIAL PRIMARY KEY,
    service_name VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    version VARCHAR(50),
    dependencies JSONB,
    metrics JSONB,
    last_check TIMESTAMP DEFAULT NOW(),
    CONSTRAINT status_check CHECK (status IN ('healthy', 'degraded', 'unhealthy'))
);

CREATE UNIQUE INDEX idx_service_health_name ON service_health(service_name);

-- API Request Metrics
CREATE TABLE IF NOT EXISTS api_metrics (
    id SERIAL PRIMARY KEY,
    service VARCHAR(50) NOT NULL,
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INT NOT NULL,
    response_time_ms INT NOT NULL,
    request_size_bytes INT,
    response_size_bytes INT,
    user_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_api_metrics_service ON api_metrics(service);
CREATE INDEX idx_api_metrics_endpoint ON api_metrics(endpoint);
CREATE INDEX idx_api_metrics_created_at ON api_metrics(created_at DESC);

-- Materialized view for mission plan statistics
CREATE MATERIALIZED VIEW mission_plan_stats AS
SELECT 
    DATE(created_at) as date,
    status,
    COUNT(*) as plan_count,
    AVG((activities::jsonb->>'length')::int) as avg_activities,
    AVG((power_summary::jsonb->>'budget_wh')::float) as avg_power_budget
FROM mission_plans
GROUP BY DATE(created_at), status;

CREATE INDEX idx_mission_plan_stats_date ON mission_plan_stats(date DESC);

-- Refresh function for materialized view
CREATE OR REPLACE FUNCTION refresh_mission_plan_stats()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mission_plan_stats;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_mission_plans_updated_at
    BEFORE UPDATE ON mission_plans
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_nasa_data_cache_updated_at
    BEFORE UPDATE ON nasa_data_cache
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Initial service health records
INSERT INTO service_health (service_name, status, version) VALUES
    ('planning-service', 'healthy', '1.0.0'),
    ('vision-service', 'healthy', '1.0.0'),
    ('marl-service', 'healthy', '1.0.0'),
    ('data-service', 'healthy', '1.0.0')
ON CONFLICT (service_name) DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;

-- Comments for documentation
COMMENT ON TABLE mission_plans IS 'Stores all generated mission plans with goals, activities, and constraints';
COMMENT ON TABLE marl_episodes IS 'Training history for multi-agent reinforcement learning system';
COMMENT ON TABLE vision_results IS 'Terrain hazard classification results from vision model';
COMMENT ON TABLE audit_logs IS 'Comprehensive audit trail for all system actions';
COMMENT ON TABLE nasa_data_cache IS 'Cached responses from NASA APIs with TTL';
COMMENT ON TABLE dem_files IS 'Registry of Digital Elevation Model files';
COMMENT ON TABLE service_health IS 'Real-time health status of all microservices';
COMMENT ON TABLE api_metrics IS 'Performance metrics for API endpoints';

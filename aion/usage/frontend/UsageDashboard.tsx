/**
 * AION Usage Dashboard Components
 *
 * State-of-the-art React components for:
 * - Usage visualization with charts
 * - Real-time usage tracking
 * - Limit warnings and alerts
 * - Subscription management UI
 */

import React, { useState, useEffect, useMemo } from 'react';
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
} from 'recharts';

// =============================================================================
// Types
// =============================================================================

interface UsageMetric {
  used: number;
  limit: number | null;
  unlimited: boolean;
  percentage: number;
  soft_limit_reached: boolean;
  hard_limit_reached: boolean;
}

interface UsageSummary {
  period: string;
  usage: Record<string, UsageMetric>;
  tier: string;
  billing_cycle_end: string | null;
  days_remaining: number;
}

interface UsageHistory {
  date: string;
  value: number;
}

interface Alert {
  alert_id: string;
  alert_type: string;
  severity: string;
  metric: string;
  current_value: number;
  limit_value: number | null;
  percentage: number;
  title: string;
  message: string;
  action_url?: string;
  action_label?: string;
}

interface Forecast {
  metric: string;
  current: number;
  limit: number | null;
  forecasted_total: number;
  will_exceed_limit: boolean;
  days_until_exceeded: number | null;
  recommended_daily_limit: number | null;
}

// =============================================================================
// Usage Bar Component
// =============================================================================

interface UsageBarProps {
  label: string;
  used: number;
  limit: number | null;
  unlimited?: boolean;
  showPercentage?: boolean;
  size?: 'sm' | 'md' | 'lg';
  colorScheme?: 'default' | 'warning' | 'danger';
}

export const UsageBar: React.FC<UsageBarProps> = ({
  label,
  used,
  limit,
  unlimited = false,
  showPercentage = true,
  size = 'md',
  colorScheme = 'default',
}) => {
  const percentage = useMemo(() => {
    if (unlimited || !limit) return 0;
    return Math.min(100, (used / limit) * 100);
  }, [used, limit, unlimited]);

  const getColorClass = () => {
    if (colorScheme === 'danger' || percentage >= 100) {
      return 'bg-red-500';
    }
    if (colorScheme === 'warning' || percentage >= 80) {
      return 'bg-yellow-500';
    }
    return 'bg-blue-500';
  };

  const heightClass = size === 'sm' ? 'h-2' : size === 'lg' ? 'h-6' : 'h-4';

  return (
    <div className="usage-bar">
      <div className="flex justify-between mb-1">
        <span className="text-sm font-medium text-gray-700">{label}</span>
        <span className="text-sm text-gray-500">
          {used.toLocaleString()}
          {!unlimited && limit && ` / ${limit.toLocaleString()}`}
          {unlimited && ' (unlimited)'}
        </span>
      </div>
      <div className={`w-full bg-gray-200 rounded-full ${heightClass}`}>
        <div
          className={`${getColorClass()} ${heightClass} rounded-full transition-all duration-300`}
          style={{ width: unlimited ? '0%' : `${percentage}%` }}
        />
      </div>
      {showPercentage && !unlimited && limit && (
        <div className="text-right text-xs text-gray-500 mt-1">
          {percentage.toFixed(1)}% used
        </div>
      )}
    </div>
  );
};

// =============================================================================
// Limit Warning Banner
// =============================================================================

interface LimitWarningProps {
  metric: string;
  percentage: number;
  limit: number;
  upgradeUrl?: string;
  onDismiss?: () => void;
}

export const LimitWarning: React.FC<LimitWarningProps> = ({
  metric,
  percentage,
  limit,
  upgradeUrl = '/pricing',
  onDismiss,
}) => {
  const getSeverity = () => {
    if (percentage >= 100) return 'critical';
    if (percentage >= 90) return 'warning';
    return 'info';
  };

  const severity = getSeverity();

  const bgColor = {
    critical: 'bg-red-100 border-red-500',
    warning: 'bg-yellow-100 border-yellow-500',
    info: 'bg-blue-100 border-blue-500',
  }[severity];

  const textColor = {
    critical: 'text-red-800',
    warning: 'text-yellow-800',
    info: 'text-blue-800',
  }[severity];

  const icon = {
    critical: '🚫',
    warning: '⚠️',
    info: 'ℹ️',
  }[severity];

  const getMessage = () => {
    if (percentage >= 100) {
      return `You've reached your ${metric} limit of ${limit.toLocaleString()}.`;
    }
    if (percentage >= 90) {
      return `You've used ${percentage.toFixed(0)}% of your ${metric} limit.`;
    }
    return `You're approaching your ${metric} limit (${percentage.toFixed(0)}% used).`;
  };

  return (
    <div className={`${bgColor} border-l-4 p-4 mb-4 rounded-r`} role="alert">
      <div className="flex items-start">
        <span className="text-2xl mr-3">{icon}</span>
        <div className="flex-1">
          <p className={`font-medium ${textColor}`}>
            {percentage >= 100 ? 'Limit Reached' : 'Usage Warning'}
          </p>
          <p className={`text-sm ${textColor} mt-1`}>{getMessage()}</p>
          {percentage >= 80 && (
            <a
              href={upgradeUrl}
              className="inline-block mt-2 px-4 py-2 bg-blue-600 text-white text-sm rounded hover:bg-blue-700"
            >
              Upgrade Plan
            </a>
          )}
        </div>
        {onDismiss && (
          <button
            onClick={onDismiss}
            className="text-gray-400 hover:text-gray-600"
          >
            ✕
          </button>
        )}
      </div>
    </div>
  );
};

// =============================================================================
// Usage Chart Component
// =============================================================================

interface UsageChartProps {
  data: UsageHistory[];
  metric: string;
  limit?: number | null;
  height?: number;
}

export const UsageChart: React.FC<UsageChartProps> = ({
  data,
  metric,
  limit,
  height = 300,
}) => {
  const formatLabel = (name: string) => {
    return name.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase());
  };

  return (
    <div className="usage-chart bg-white rounded-lg shadow p-4">
      <h3 className="text-lg font-semibold mb-4">{formatLabel(metric)} Usage</h3>
      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="date"
            tick={{ fontSize: 12 }}
            tickFormatter={(value) => {
              const date = new Date(value);
              return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
            }}
          />
          <YAxis tick={{ fontSize: 12 }} />
          <Tooltip
            labelFormatter={(label) => new Date(label).toLocaleDateString()}
            formatter={(value: number) => [value.toLocaleString(), formatLabel(metric)]}
          />
          <Area
            type="monotone"
            dataKey="value"
            stroke="#3B82F6"
            fill="#93C5FD"
            strokeWidth={2}
          />
          {limit && (
            <Area
              type="monotone"
              dataKey={() => limit}
              stroke="#EF4444"
              strokeDasharray="5 5"
              fill="none"
              name="Limit"
            />
          )}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};

// =============================================================================
// Usage Breakdown Chart
// =============================================================================

interface BreakdownChartProps {
  data: Record<string, number>;
  title: string;
}

const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899'];

export const BreakdownChart: React.FC<BreakdownChartProps> = ({ data, title }) => {
  const chartData = useMemo(() => {
    return Object.entries(data).map(([name, value]) => ({
      name: name.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase()),
      value,
    }));
  }, [data]);

  return (
    <div className="breakdown-chart bg-white rounded-lg shadow p-4">
      <h3 className="text-lg font-semibold mb-4">{title}</h3>
      <ResponsiveContainer width="100%" height={250}>
        <PieChart>
          <Pie
            data={chartData}
            cx="50%"
            cy="50%"
            labelLine={false}
            outerRadius={80}
            fill="#8884d8"
            dataKey="value"
            label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}
          >
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip formatter={(value: number) => value.toLocaleString()} />
          <Legend />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
};

// =============================================================================
// Tier Badge Component
// =============================================================================

interface TierBadgeProps {
  tier: string;
  size?: 'sm' | 'md' | 'lg';
}

export const TierBadge: React.FC<TierBadgeProps> = ({ tier, size = 'md' }) => {
  const config = {
    free: { bg: 'bg-gray-100', text: 'text-gray-800', label: 'Free' },
    pro: { bg: 'bg-blue-100', text: 'text-blue-800', label: 'Pro' },
    executive: { bg: 'bg-purple-100', text: 'text-purple-800', label: 'Executive' },
    team: { bg: 'bg-green-100', text: 'text-green-800', label: 'Team' },
    enterprise: { bg: 'bg-yellow-100', text: 'text-yellow-800', label: 'Enterprise' },
  }[tier.toLowerCase()] || { bg: 'bg-gray-100', text: 'text-gray-800', label: tier };

  const sizeClass = {
    sm: 'text-xs px-2 py-0.5',
    md: 'text-sm px-3 py-1',
    lg: 'text-base px-4 py-2',
  }[size];

  return (
    <span className={`${config.bg} ${config.text} ${sizeClass} rounded-full font-medium`}>
      {config.label}
    </span>
  );
};

// =============================================================================
// Forecast Card Component
// =============================================================================

interface ForecastCardProps {
  forecast: Forecast;
}

export const ForecastCard: React.FC<ForecastCardProps> = ({ forecast }) => {
  return (
    <div className="forecast-card bg-white rounded-lg shadow p-4">
      <h4 className="text-sm font-medium text-gray-500 mb-2">
        {forecast.metric.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())} Forecast
      </h4>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="text-2xl font-bold text-gray-900">
            {forecast.forecasted_total.toLocaleString()}
          </p>
          <p className="text-sm text-gray-500">Projected end-of-month</p>
        </div>
        {forecast.limit && (
          <div>
            <p className="text-2xl font-bold text-gray-900">
              {forecast.limit.toLocaleString()}
            </p>
            <p className="text-sm text-gray-500">Monthly limit</p>
          </div>
        )}
      </div>
      {forecast.will_exceed_limit && (
        <div className="mt-4 p-3 bg-red-50 rounded border border-red-200">
          <p className="text-sm text-red-800">
            ⚠️ At current rate, you'll exceed your limit
            {forecast.days_until_exceeded !== null &&
              ` in ${forecast.days_until_exceeded} days`}
            .
          </p>
          {forecast.recommended_daily_limit !== null && (
            <p className="text-sm text-red-700 mt-1">
              Recommended daily limit: {forecast.recommended_daily_limit.toLocaleString()}
            </p>
          )}
        </div>
      )}
    </div>
  );
};

// =============================================================================
// Upgrade Prompt Component
// =============================================================================

interface UpgradePromptProps {
  currentTier: string;
  reason?: string;
  onUpgrade?: () => void;
}

export const UpgradePrompt: React.FC<UpgradePromptProps> = ({
  currentTier,
  reason,
  onUpgrade,
}) => {
  const nextTier = {
    free: 'Pro',
    pro: 'Executive',
    executive: 'Team',
    team: 'Enterprise',
  }[currentTier.toLowerCase()] || 'Pro';

  const benefits = {
    Pro: [
      'Unlimited messages',
      'Access to all 34 experts',
      '1,000 API calls/month',
      'Permanent memory retention',
    ],
    Executive: [
      'Priority processing',
      '10,000 API calls/month',
      'Custom expert training',
      'Dedicated support',
    ],
    Team: [
      'Multi-seat access',
      'Shared memory pool',
      'Admin controls',
      'Team analytics',
    ],
  }[nextTier] || [];

  return (
    <div className="upgrade-prompt bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg p-6 text-white">
      <h3 className="text-xl font-bold mb-2">Upgrade to {nextTier}</h3>
      {reason && <p className="text-blue-100 mb-4">{reason}</p>}
      <ul className="mb-4 space-y-2">
        {benefits.map((benefit, index) => (
          <li key={index} className="flex items-center">
            <span className="mr-2">✓</span>
            {benefit}
          </li>
        ))}
      </ul>
      <button
        onClick={onUpgrade}
        className="w-full bg-white text-blue-600 px-6 py-3 rounded-lg font-semibold hover:bg-blue-50 transition"
      >
        Upgrade Now
      </button>
    </div>
  );
};

// =============================================================================
// Main Usage Dashboard Component
// =============================================================================

interface UsageDashboardProps {
  userId?: string;
  apiBaseUrl?: string;
}

export const UsageDashboard: React.FC<UsageDashboardProps> = ({
  userId,
  apiBaseUrl = '/api/v1/usage',
}) => {
  const [summary, setSummary] = useState<UsageSummary | null>(null);
  const [history, setHistory] = useState<Record<string, UsageHistory[]>>({});
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedMetric, setSelectedMetric] = useState<string>('messages_sent');

  // Fetch usage data
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);

        // Fetch current usage
        const summaryRes = await fetch(`${apiBaseUrl}/current`);
        if (!summaryRes.ok) throw new Error('Failed to fetch usage');
        const summaryData = await summaryRes.json();
        setSummary(summaryData);

        // Fetch alerts
        const alertsRes = await fetch(`${apiBaseUrl}/alerts`);
        if (alertsRes.ok) {
          const alertsData = await alertsRes.json();
          setAlerts(alertsData);
        }

        // Fetch history for selected metric
        const historyRes = await fetch(
          `${apiBaseUrl}/history?metric=${selectedMetric}&days=30`
        );
        if (historyRes.ok) {
          const historyData = await historyRes.json();
          setHistory((prev) => ({
            ...prev,
            [selectedMetric]: historyData.history,
          }));
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [apiBaseUrl, selectedMetric]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
        <p className="font-bold">Error</p>
        <p>{error}</p>
      </div>
    );
  }

  if (!summary) return null;

  const metrics = Object.entries(summary.usage);

  return (
    <div className="usage-dashboard space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Usage Dashboard</h2>
          <p className="text-gray-500">
            Billing period: {summary.period} ({summary.days_remaining} days remaining)
          </p>
        </div>
        <TierBadge tier={summary.tier} size="lg" />
      </div>

      {/* Alerts */}
      {alerts.map((alert) => (
        <LimitWarning
          key={alert.alert_id}
          metric={alert.metric}
          percentage={alert.percentage}
          limit={alert.limit_value || 0}
          onDismiss={() => setAlerts((a) => a.filter((x) => x.alert_id !== alert.alert_id))}
        />
      ))}

      {/* Usage Bars */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Current Usage</h3>
        <div className="space-y-4">
          {metrics.map(([name, metric]) => (
            <UsageBar
              key={name}
              label={name.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())}
              used={metric.used}
              limit={metric.limit}
              unlimited={metric.unlimited}
              colorScheme={
                metric.hard_limit_reached
                  ? 'danger'
                  : metric.soft_limit_reached
                  ? 'warning'
                  : 'default'
              }
            />
          ))}
        </div>
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* History Chart */}
        <div className="bg-white rounded-lg shadow">
          <div className="p-4 border-b">
            <select
              value={selectedMetric}
              onChange={(e) => setSelectedMetric(e.target.value)}
              className="border rounded px-3 py-2"
            >
              {metrics.map(([name]) => (
                <option key={name} value={name}>
                  {name.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())}
                </option>
              ))}
            </select>
          </div>
          {history[selectedMetric] && (
            <UsageChart
              data={history[selectedMetric]}
              metric={selectedMetric}
              limit={summary.usage[selectedMetric]?.limit}
            />
          )}
        </div>

        {/* Upgrade Prompt (for free tier) */}
        {summary.tier === 'free' && (
          <UpgradePrompt
            currentTier={summary.tier}
            reason="Unlock unlimited messages and all experts"
            onUpgrade={() => (window.location.href = '/pricing')}
          />
        )}
      </div>
    </div>
  );
};

export default UsageDashboard;

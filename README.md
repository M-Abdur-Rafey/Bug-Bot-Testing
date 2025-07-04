# Real-Time Translator Extension Server

A WebSocket-based real-time translation server that converts speech to text and translates it using multiple AI providers including OpenAI GPT-4, Claude, and Deepgram's speech recognition services.

## Features

- Real-time Speech Recognition using Deepgram's Nova-2 model
- Multi-provider Translation Support:
  - **OpenAI**: GPT-4 Realtime API and REST API
  - **Claude**: Anthropic's Claude 3.5 Sonnet with streaming
  - **DeepGram**: Basic translation capabilities (expandable)
- Multi-language Support
- Two Translation Modes:
  - Speed Mode: Translates each sentence as it's finalized
  - Accuracy Mode: Waits for complete utterances before translation
- WebSocket Communication
- CORS Enabled
- Health Check Endpoint

## Prerequisites

- Python 3.8+
- Deepgram API key (for speech recognition)
- At least one of the following for translation:
  - OpenAI API key
  - Claude (Anthropic) API key
  - DeepGram API key

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd Translator-Extension-Server
   ```

2. Create and activate virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create `.env` file with required environment variables:

   ```env
   # API Keys for various services
   DEEPGRAM_API_KEY=your_deepgram_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   CLAUDE_API_KEY=your_claude_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here

   # JWT Configuration
   JWT_SECRET_KEY=your_jwt_secret_key_here

   # Stripe Configuration (Required for payment processing)
   STRIPE_SECRET_KEY=sk_test_your_stripe_secret_key_here
   STRIPE_PUBLISHABLE_KEY=pk_test_your_stripe_publishable_key_here
   STRIPE_WEBHOOK_SECRET=whsec_your_webhook_secret_here

   # Database Configuration (if needed)
   MONGODB_URI=mongodb://localhost:27017/translator_db
   ```

## Usage

Start the server:

```bash
uvicorn main:app --reload --port 7015
```

Connect to WebSocket with translation provider:

```
ws://localhost:7015/listen?source=English&target=French&mode=speed&provider=openai
```

### Translation Providers

#### OpenAI

```
ws://localhost:7015/listen?provider=openai&source=English&target=French&mode=speed
```

- Uses GPT-4o for translation
- Supports both realtime WebSocket and REST API
- High quality translations
- Streaming responses

#### Claude (Anthropic)

```
ws://localhost:7015/listen?provider=claude&source=English&target=French&mode=speed
```

- Uses Claude 3.5 Sonnet for translation
- REST API with streaming support
- Excellent for complex translations
- Streaming responses

#### DeepGram

```
ws://localhost:7015/listen?provider=deepgram&source=English&target=French&mode=speed
```

- Basic translation capabilities
- Fast response times
- Can be extended with other translation services
- Streaming responses

## File Structure

- `main.py`: FastAPI application setup, WebSocket endpoint, and health check
- `assistant.py`: Core translation logic with multi-provider support
- `languages.py`: Language codes mapping for Deepgram
- `Authentication.py`: API key verification and authentication
- `requirements.txt`: Python dependencies
- `run.sh`: Production deployment script

## Environment Variables

Required environment variables:

- `DEEPGRAM_API_KEY`: Your Deepgram API key for speech recognition

Optional environment variables (based on chosen translation provider):

- `OPENAI_API_KEY`: Your OpenAI API key (can also be passed via WebSocket)
- `CLAUDE_API_KEY`: Your Claude/Anthropic API key (can also be passed via WebSocket)
- `GEMINI_API_KEY`: Your Gemini API Key

## WebSocket Authorization

Send an authorization message after connecting:

```javascript
{
  "type": "authorization",
  "openai_api_key": "your_openai_key_here",  // Required for OpenAI provider
  "claude_api_key": "your_claude_key_here"   // Required for Claude provider
}
```

## Database Schema

The application uses MongoDB to store user data and translation sessions. Here's the schema structure:

```json
{
  "Name": "String",
  "Email": "String",
  "Password": "String",
  "Created_At": "Date (YYYY-MM-DD)",
  "Summary and Keywords": Boolean,
  "paid_user":Boolean,
  "Credits": double,
  "API_Key": {
    "OpenAI": "String",
    "Deepgram": "String",
    "Claude": "String"
  },
  "Session": {
    "YYYY-MM-DD": {
      "Youtube": {
        "timestamp": {
          "Original Text": "String",
          "Translation": "String",
          "summary": "String",
          "orginal language": "String",
          "translation language": "String",
          "Keywords": list
        }
      },
      "Google Meet": {
        "timestamp": {
          "Original Text": "String",
          "Translation": "String",
          "summary": "String",
          "orginal language": "String",
          "translation language": "String",
          "Keywords": list
        }
      }
    }
  }
}
```

### Schema Details

- **User Information**:

  - `Name`: User's full name
  - `Email`: User's email address
  - `Password`: Hashed password
  - `Created_At`: Account creation date
  - `Summary and Keywords`: Does user wish to have keyword and sumamry feature

- **API Keys**:

  - Stores API keys for different services (OpenAI, Deepgram, Claude)

- **Sessions**:
  - Organized by date (YYYY-MM-DD)
  - Each date contains different platforms (Youtube, Google Meet)
  - Each platform entry contains:
    - Timestamp of translation
    - Original text
    - Translated text
    - Summary
    - Source language
    - Target language

## API Endpoints

### Core Endpoints

- `GET /health`: Health check endpoint
- `WebSocket /listen`: Main WebSocket endpoint for real-time translation

### Authentication Endpoints

- `POST /register`: User registration
- `POST /login`: User login
- `POST /auth/google`: Google OAuth authentication
- `POST /logout`: User logout
- `POST /verify-token`: Token verification

### Payment & Credit Endpoints

- `GET /get-packages`: Get available subscription packages
- `POST /stripe/webhook`: Stripe webhook for payment processing
- `POST /stripe/create-payment-intent`: Create payment intent for custom flows
- `POST /stripe/purchase-credits`: Create payment intent for specific credit purchases from frontend
- `POST /stripe/create-checkout-session`: Create Stripe checkout session for package purchases
- `GET /stripe/payment-status/{payment_intent_id}`: Get payment status
- `GET /stripe/packages`: Get formatted credit packages for frontend pricing page
- `GET /get-credits`: Get user's current credit balance
- `POST /add-credits`: Manually add credits to user account
- `GET /credit-history`: Get user's credit transaction history
- `POST /check-sufficient-credits`: Check if user has sufficient credits

### User Management Endpoints

- `GET /Get-User-Data`: Get user profile data
- `POST /update-API-Key`: Update user's API keys
- `POST /get-api-key`: Get user's API keys
- `POST /Check-Payment-Status`: Check payment status
- `POST /update-paid-user`: Update user's paid status

### WebSocket Parameters

- `source`: Source language (default: English)
- `target`: Target language (default: French)
- `mode`: Translation mode - 'speed' or 'accuracy' (default: speed)
- `provider`: Translation provider - 'openai', 'claude', or 'deepgram' (default: openai)

## Translation Modes

1. Speed Mode (`mode=speed`):

   - Translates each finalized sentence immediately
   - Lower latency but may translate incomplete thoughts
   - Best for real-time conversations

2. Accuracy Mode (`mode=accuracy`):
   - Waits for complete utterances before translation
   - Higher accuracy but slightly higher latency
   - Best for formal presentations or speeches

## Provider Comparison

| Provider | Strengths                         | Best For                                | API Type         |
| -------- | --------------------------------- | --------------------------------------- | ---------------- |
| OpenAI   | High quality, multiple models     | General purpose, technical content      | WebSocket + REST |
| Claude   | Excellent reasoning, long context | Complex translations, formal text       | REST (streaming) |
| DeepGram | Fast, extensible                  | Basic translations, custom integrations | REST             |

## Stripe Payment Integration

The application includes a complete Stripe payment integration for processing user subscriptions and credit purchases.

### Stripe Setup

1. **Create a Stripe Account**: Sign up at [https://stripe.com](https://stripe.com)

2. **Get API Keys**: Navigate to the Stripe Dashboard → Developers → API keys

   - Copy the Secret key (starts with `sk_test_` for test mode)
   - Copy the Publishable key (starts with `pk_test_` for test mode)

3. **Configure Webhook Endpoint**:

   - Go to Stripe Dashboard → Developers → Webhooks
   - Click "Add endpoint"
   - URL: `https://your-domain.com/stripe/webhook`
   - Select events to listen for:
     - `payment_intent.succeeded`
     - `checkout.session.completed`
     - `payment_intent.payment_failed`
   - Copy the webhook signing secret (starts with `whsec_`)

4. **Environment Variables**: Add these to your `.env` file:
   ```env
   STRIPE_SECRET_KEY=sk_test_your_secret_key_here
   STRIPE_PUBLISHABLE_KEY=pk_test_your_publishable_key_here
   STRIPE_WEBHOOK_SECRET=whsec_your_webhook_secret_here
   ```

### Payment Packages

The application supports three subscription tiers defined in `Packages.py`:

- **Basic Package**: $10 - 1,000 credits
- **Pro Package**: $40 - 5,000 credits
- **Enterprise Package**: $70 - 10,000 credits

Each package includes Stripe payment links that redirect users to hosted checkout pages.

### Webhook Flow

1. User completes payment on Stripe-hosted checkout
2. Stripe sends webhook event to `/stripe/webhook` endpoint
3. Server verifies webhook signature for security
4. Credits are automatically added to user's account
5. User's payment status is updated in the database
6. Transaction is logged for audit purposes

### Testing Payments

Use Stripe's test card numbers for testing:

- **Success**: `4242 4242 4242 4242`
- **Failure**: `4000 0000 0000 0002`
- **3D Secure**: `4000 0025 0000 3155`

### API Usage Examples

```javascript
// Create a payment intent
const response = await fetch("/stripe/create-payment-intent", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    Authorization: `Bearer ${userToken}`,
  },
  body: JSON.stringify({
    amount: 1000, // Amount in cents ($10.00)
    currency: "usd",
  }),
});

// Check payment status
const statusResponse = await fetch(
  `/stripe/payment-status/${paymentIntentId}`,
  {
    headers: {
      Authorization: `Bearer ${userToken}`,
    },
  }
);
```

## Examples

### Using with OpenAI

```javascript
// Connect
const ws = new WebSocket(
  "ws://localhost:7015/listen?provider=openai&source=English&target=French"
);

// Authorize
ws.send(
  JSON.stringify({
    type: "authorization",
    openai_api_key: "sk-...",
  })
);
```

### Using with Claude

```javascript
// Connect
const ws = new WebSocket(
  "ws://localhost:7015/listen?provider=claude&source=English&target=Spanish"
);

// Authorize
ws.send(
  JSON.stringify({
    type: "authorization",
    claude_api_key: "sk-ant-...",
  })
);
```

### Using with DeepGram

```javascript
// Connect
const ws = new WebSocket(
  "ws://localhost:7015/listen?provider=deepgram&source=English&target=German"
);

// Authorize (DeepGram key from environment)
ws.send(
  JSON.stringify({
    type: "authorization",
  })
);
```

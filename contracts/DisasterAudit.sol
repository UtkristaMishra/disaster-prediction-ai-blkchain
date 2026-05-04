// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DisasterAudit {
    struct AuditEntry {
        uint256 timestamp;
        string ipfsHash;
        string metadata;
        uint256 riskScore;
    }

    AuditEntry[] public entries;
    address public owner;

    event EntryLogged(uint256 indexed id, string ipfsHash, uint256 riskScore, string metadata);

    constructor() {
        owner = msg.sender;
    }

    function logPrediction(string memory ipfsHash, string memory metadata, uint256 riskScore) public {
        require(bytes(ipfsHash).length > 0, "IPFS hash is required");
        require(riskScore <= 100, "Risk score must be between 0 and 100");

        entries.push(AuditEntry(block.timestamp, ipfsHash, metadata, riskScore));
        emit EntryLogged(entries.length - 1, ipfsHash, riskScore, metadata);
    }

    function getEntry(uint256 index) public view returns (AuditEntry memory) {
        require(index < entries.length, "Entry does not exist");
        return entries[index];
    }

    function totalEntries() public view returns (uint256) {
        return entries.length;
    }
}
